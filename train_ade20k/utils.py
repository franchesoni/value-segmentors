print("Importing standard...")
import subprocess
import shutil
from pathlib import Path

print("Importing external...")
import torch
import numpy as np
from PIL import Image

REDUCTION = "none"
if REDUCTION == "umap":
    from umap import UMAP
elif REDUCTION == "tsne":
    from sklearn.manifold import TSNE
elif REDUCTION == "pca":
    from sklearn.decomposition import PCA


def symlog(x):
    return torch.sign(x) * torch.log(torch.abs(x) + 1)


def preprocess_masks_features(masks, features):
    # Get shapes right
    B, M, H, W = masks.shape
    Bf, F, Hf, Wf = features.shape
    masks = masks.reshape(B, M, 1, H * W)
    # # the following assertions should work, remove due to speed
    # assert H == Hf and W == Wf and B == Bf
    # assert masks.dtype == torch.bool
    # assert (mask_areas > 0).all(), "you shouldn't have empty masks"

    # Reduce M if there are empty masks
    mask_areas = masks.sum(dim=3)  # B, M, 1
    features = features.reshape(B, 1, F, H * W)
    # output shapes
    # features: B, 1, F, H*W
    # masks: B, M, 1, H*W

    return masks, features, M, B, H, W, F


def get_row_col(H, W, device):
    # get position of pixels in [0, 1]
    row = torch.linspace(0, 1, H, device=device)
    col = torch.linspace(0, 1, W, device=device)
    return row, col


def get_current_git_commit():
    try:
        # Run the git command to get the current commit hash
        commit_hash = subprocess.check_output(["git", "rev-parse", "HEAD"]).strip()
        # Decode from bytes to a string
        return commit_hash.decode("utf-8")
    except subprocess.CalledProcessError:
        # Handle the case where the command fails (e.g., not a Git repository)
        print("An error occurred while trying to retrieve the git commit hash.")
        return None


def clean_dir(dirname):
    """Removes all directories in dirname that don't have a done.txt file"""
    dstdir = Path(dirname)
    dstdir.mkdir(exist_ok=True, parents=True)
    for f in dstdir.iterdir():
        # if the directory doesn't have a done.txt file remove it
        if f.is_dir() and not (f / "done.txt").exists():
            shutil.rmtree(f)


def save_tensor_as_image(tensor, dstfile, global_step):
    dstfile = Path(dstfile)
    dstfile = (dstfile.parent / (dstfile.stem + "_" + str(global_step))).with_suffix(
        ".jpg"
    )
    save(tensor, str(dstfile))


def minmaxnorm(x):
    return (x - x.min()) / (x.max() - x.min())


def save(tensor, name, channel_offset=0):
    tensor = to_img(tensor, channel_offset=channel_offset)
    Image.fromarray(tensor).save(name)


def to_img(tensor, channel_offset=0):
    tensor = minmaxnorm(tensor)
    tensor = (tensor * 255).to(torch.uint8)
    C, H, W = tensor.shape
    if tensor.shape[0] == 1:
        tensor = tensor[0]
    elif tensor.shape[0] == 2:
        tensor = torch.stack([tensor[0], torch.zeros_like(tensor[0]), tensor[1]], dim=0)
        tensor = tensor.permute(1, 2, 0)
    elif tensor.shape[0] >= 3:
        tensor = tensor[channel_offset : channel_offset + 3]
        tensor = tensor.permute(1, 2, 0)
    tensor = tensor.cpu().numpy()
    return tensor


def log_input_output(
    name,
    x,
    y_hat,
    global_step,
    img_dstdir,
    out_dstdir,
    reduce_dim=True,
    reduction=REDUCTION,
    resample_size=20000,
):
    y_hat = y_hat.reshape(
        y_hat.shape[0], y_hat.shape[2], y_hat.shape[3], y_hat.shape[4]
    )
    if reduce_dim and y_hat.shape[1] >= 3:
        reducer = (
            UMAP(n_components=3)
            if (reduction == "umap")
            else (
                TSNE(n_components=3)
                if reduction == "tsne"
                else PCA(n_components=3) if reduction == "pca" else None
            )
        )
        np_y_hat = y_hat.detach().cpu().permute(1, 0, 2, 3).numpy()  # F, 1, B, H, W
        np_y_hat = np_y_hat.reshape(np_y_hat.shape[0], -1)  # F, BHW
        np_y_hat = np_y_hat.T  # BHW, F
        sampled_pixels = np_y_hat[:: np_y_hat.shape[0] // resample_size]
        print("dim reduction fit..." + " " * 30, end="\r")
        reducer = reducer.fit(sampled_pixels)
        print("dim reduction transform..." + " " * 30, end="\r")
        reducer.transform(np_y_hat[:10])  # to numba compile the function
        np_y_hat = reducer.transform(np_y_hat)  # BHW, 3
        # revert back to original shape
        y_hat2 = (
            torch.from_numpy(
                np_y_hat.T.reshape(3, y_hat.shape[0], y_hat.shape[2], y_hat.shape[3])
            )
            .to(y_hat.device)
            .permute(1, 0, 2, 3)
        )
        print("done" + " " * 30, end="\r")
    else:
        y_hat2 = y_hat

    for i in range(min(len(x), 8)):
        save_tensor_as_image(
            x[i],
            img_dstdir / f"input_{name}_{str(i).zfill(2)}",
            global_step=global_step,
        )
        for c in range(y_hat.shape[1]):
            save_tensor_as_image(
                y_hat[i, c : c + 1],
                out_dstdir / f"pred_channel_{name}_{str(i).zfill(2)}_{c}",
                global_step=global_step,
            )
        # log color image

        assert len(y_hat2.shape) == 4, "should be B, F, H, W"
        if reduce_dim:
            save_tensor_as_image(
                y_hat2[i][:3],
                out_dstdir / f"pred_reduced_{name}_{str(i).zfill(2)}",
                global_step=global_step,
            )
        save_tensor_as_image(
            y_hat[i][:3],
            out_dstdir / f"pred_colorchs_{name}_{str(i).zfill(2)}",
            global_step=global_step,
        )


def check_for_nan(loss, model, batch):
    try:
        assert torch.isnan(loss) == False
    except Exception as e:
        # print things useful to debug
        # does the batch contain nan?
        print("img batch contains nan?", torch.isnan(batch[0]).any())
        print("mask batch contains nan?", torch.isnan(batch[1]).any())
        # does the model weights contain nan?
        for name, param in model.named_parameters():
            if torch.isnan(param).any():
                print(name, "contains nan")
        # does the output contain nan?
        print("output contains nan?", torch.isnan(model(batch[0])).any())
        # now raise the error
        raise e


def calculate_iou(pred, label):
    intersection = ((label == 1) & (pred == 1)).sum()
    union = ((label == 1) | (pred == 1)).sum()
    if not union:
        return 0
    else:
        iou = intersection.item() / union.item()
        return iou


def load_from_ckpt(net, ckpt_path, strict=True):
    """Load network weights"""
    if ckpt_path and Path(ckpt_path).exists():
        ckpt = torch.load(ckpt_path, map_location="cpu")
        if "MODEL_STATE" in ckpt:
            ckpt = ckpt["MODEL_STATE"]
        elif "state_dict" in ckpt:
            ckpt = ckpt["state_dict"]
        net.load_state_dict(ckpt, strict=strict)
        print("Loaded checkpoint from", ckpt_path)
    return net


def seed_everything(seed):
    """Seed everything"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def print_memory_usage(local_vars):
    print("Memory usage (approximate):")
    for var_name, value in local_vars.items():
        if torch.is_tensor(value):
            size = value.element_size() * value.nelement() / 1024
            print(f"Tensor {var_name}: {size:.2f} KiB")
        else:
            size = sys.getsizeof(value) / 1024
            print(f"{var_name}: {size:.2f} KiB")
