print("Importing external...")
import torch
from torch import nn
import torch.nn.functional as F

from timm.models.efficientvit_mit import (
    ConvNormAct,
    FusedMBConv,
    MBConv,
    ResidualBlock,
    efficientvit_l2,
)
from timm.layers import GELUTanh


def val2list(x: list or tuple or any, repeat_time=1):
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x for _ in range(repeat_time)]


def resize(
    x: torch.Tensor,
    size: any or None = None,
    scale_factor: list[float] or None = None,
    mode: str = "bicubic",
    align_corners: bool or None = False,
) -> torch.Tensor:
    if mode in {"bilinear", "bicubic"}:
        return F.interpolate(
            x,
            size=size,
            scale_factor=scale_factor,
            mode=mode,
            align_corners=align_corners,
        )
    elif mode in {"nearest", "area"}:
        return F.interpolate(x, size=size, scale_factor=scale_factor, mode=mode)
    else:
        raise NotImplementedError(f"resize(mode={mode}) not implemented.")


class UpSampleLayer(nn.Module):
    def __init__(
        self,
        mode="bicubic",
        size: int or tuple[int, int] or list[int] or None = None,
        factor=2,
        align_corners=False,
    ):
        super(UpSampleLayer, self).__init__()
        self.mode = mode
        self.size = val2list(size, 2) if size is not None else None
        self.factor = None if self.size is not None else factor
        self.align_corners = align_corners

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (
            self.size is not None and tuple(x.shape[-2:]) == self.size
        ) or self.factor == 1:
            return x
        return resize(x, self.size, self.factor, self.mode, self.align_corners)


class DAGBlock(nn.Module):
    def __init__(
        self,
        inputs: dict[str, nn.Module],
        merge: str,
        post_input: nn.Module or None,
        middle: nn.Module,
        outputs: dict[str, nn.Module],
    ):
        super(DAGBlock, self).__init__()

        self.input_keys = list(inputs.keys())
        self.input_ops = nn.ModuleList(list(inputs.values()))
        self.merge = merge
        self.post_input = post_input

        self.middle = middle

        self.output_keys = list(outputs.keys())
        self.output_ops = nn.ModuleList(list(outputs.values()))

    def forward(self, feature_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        feat = [
            op(feature_dict[key]) for key, op in zip(self.input_keys, self.input_ops)
        ]
        if self.merge == "add":
            feat = list_sum(feat)
        elif self.merge == "cat":
            feat = torch.concat(feat, dim=1)
        else:
            raise NotImplementedError
        if self.post_input is not None:
            feat = self.post_input(feat)
        feat = self.middle(feat)
        for key, op in zip(self.output_keys, self.output_ops):
            feature_dict[key] = op(feat)
        return feature_dict


def list_sum(x: list) -> any:
    return x[0] if len(x) == 1 else x[0] + list_sum(x[1:])


class SegHead(nn.Module):
    def __init__(
        self,
        fid_list: list[str],
        in_channel_list: list[int],
        stride_list: list[int],
        head_stride: int,
        head_width: int,
        head_depth: int,
        expand_ratio: float,
        middle_op: str,
        final_expand: float or None,
        n_classes: int,
        dropout=0,
        norm="bn2d",
        act_func="hswish",
    ):
        super(SegHead, self).__init__()
        # exceptions to adapt effvit to timm
        if act_func == "gelu":
            act_func = GELUTanh
        else:
            raise ValueError(f"act_func {act_func} not supported")
        if norm == "bn2d":
            norm_layer = nn.BatchNorm2d
        else:
            raise ValueError(f"norm {norm} not supported")

        inputs = {}
        for fid, in_channel, stride in zip(fid_list, in_channel_list, stride_list):
            factor = stride // head_stride
            if factor == 1:
                inputs[fid] = ConvNormAct(
                    in_channel, head_width, 1, norm_layer=norm_layer, act_layer=act_func
                )
            else:
                inputs[fid] = nn.Sequential(
                    ConvNormAct(
                        in_channel,
                        head_width,
                        1,
                        norm_layer=norm_layer,
                        act_layer=act_func,
                    ),
                    UpSampleLayer(factor=factor),
                )
        self.in_keys = inputs.keys()
        self.in_ops = nn.ModuleList(inputs.values())

        middle = []
        for _ in range(head_depth):
            if middle_op == "mbconv":
                block = MBConv(
                    head_width,
                    head_width,
                    expand_ratio=expand_ratio,
                    norm_layer=norm_layer,
                    act_layer=(act_func, act_func, None),
                )
            elif middle_op == "fmbconv":
                block = FusedMBConv(
                    head_width,
                    head_width,
                    expand_ratio=expand_ratio,
                    norm_layer=norm_layer,
                    act_layer=(act_func, None),
                )
            else:
                raise NotImplementedError
            middle.append(ResidualBlock(block, nn.Identity()))
        self.middle = nn.Sequential(*middle)

        self.out_layer = nn.Sequential(
            *[
                (
                    None
                    if final_expand is None
                    else ConvNormAct(
                        head_width,
                        head_width * final_expand,
                        1,
                        norm_layer=norm_layer,
                        act_layer=act_func,
                    )
                ),
                ConvNormAct(
                    head_width * (final_expand or 1),
                    n_classes,
                    1,
                    bias=True,
                    dropout=dropout,
                    norm_layer=None,
                    act_layer=None,
                ),
            ]
        )

    def forward(self, feature_map_list):
        t_feat_maps = [
            self.in_ops[ind](feature_map_list[ind])
            for ind in range(len(feature_map_list))
        ]
        t_feat_map = list_sum(t_feat_maps)
        t_feat_map = self.middle(t_feat_map)
        out = self.out_layer(t_feat_map)
        return out


class EfficientViTSegL2(nn.Module):
    def __init__(self, use_norm_params=False, pretrained=True):
        super(EfficientViTSegL2, self).__init__()
        self.bbone = efficientvit_l2(
            num_classes=0, features_only=True, pretrained=pretrained
        )
        self.head = SegHead(
            fid_list=["stage4", "stage3", "stage2"],
            in_channel_list=[512, 256, 128],
            stride_list=[32, 16, 8],
            head_stride=8,
            head_width=128,
            head_depth=3,
            expand_ratio=4,
            middle_op="fmbconv",
            final_expand=8,
            n_classes=150,
            act_func="gelu",
        )

        # [optional] deactivate normalization
        if not use_norm_params:
            for module in self.modules():
                if (
                    isinstance(module, nn.LayerNorm)
                    or isinstance(module, nn.BatchNorm2d)
                    or isinstance(module, nn.BatchNorm1d)
                ):
                    module.weight.requires_grad_(False)
                    module.bias.requires_grad_(False)

    def forward(self, x):
        feat = self.bbone(x)
        out = self.head([feat[3], feat[2], feat[1]])
        return out
