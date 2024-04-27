import gc
import os
import time
import sys
import json
import copy
from pathlib import Path
import torch
from torch.utils.tensorboard import SummaryWriter

from network import EfficientViTSegL2
from utils import (
    clean_dir,
    get_current_git_commit,
    log_input_output,
    check_for_nan,
    load_from_ckpt,
    seed_everything,
)
from sing import SING
from dataloading import ADE20KDataset, train_transform, val_transform, resize
from torchvision.transforms.v2 import RandomResizedCrop, RandomPhotometricDistort, Normalize




def train(
    seed=0,
    epochs=8,
    from_scratch=False,
    use_norm_params=False,
    train_size=int(1e100),
    val_size=100,
    batch_size=46,
    num_workers=10,
    prefetch_factor=2,
    ckpt_path=None,
    gpu_number=0,
    max_lr=0.01,
    val_every_pct=0.02,
    clean=False,
    dev=False,
    tag="",
    snapshot_path="runs/snapshot.pth",
    resume=False,
):
    MAIN_GPU_NUMBER = gpu_number  # use only one gpu
    assert gpu_number == MAIN_GPU_NUMBER

    print(gpu_number, "Launched!")
    if dev:
        num_workers = 0
        from_scratch = True
    seed_everything(seed)

    ################## LOGGING ##################
    if gpu_number == MAIN_GPU_NUMBER:
        print(gpu_number, "Setting hparams...")
        hparams1 = copy.deepcopy(locals())
        command = " ".join(sys.argv)
        hparams2 = dict(
            model="effvitl1",
            git_commit=get_current_git_commit(),
            script_file=Path(__file__).name,
        )
        hparams = {"command": command, **hparams1, **hparams2}

        print(gpu_number, "Setting logger...")
        if clean:  # remove unfinished runs
            clean_dir("runs")
        logger = SummaryWriter(comment=tag)  # to save values
        img_dstdir = Path(logger.get_logdir()) / "images"  # where to save images
        out_dstdir = Path(logger.get_logdir()) / "outputs"
        img_dstdir.mkdir(exist_ok=True, parents=True)
        out_dstdir.mkdir(exist_ok=True, parents=True)
        # log hparams
        with open(Path(logger.get_logdir()) / "hparams.txt", "w") as f:
            json.dump(hparams, f, indent=4)

    ################## DATA ##################
    print(gpu_number, "Setting dataloaders...")
    train_ds = ADE20KDataset(split="training", transform=train_transform)
    val_ds = ADE20KDataset(split="validation", transform=val_transform)
    custom_collate = None
    train_ds.sample_paths = train_ds.sample_paths[:train_size]
    val_ds.sample_paths = val_ds.sample_paths[:val_size]
    train_dl = torch.utils.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=num_workers > 0,
        # persistent_workers=num_workers > 0,
        collate_fn=custom_collate,
        # prefetch_factor=prefetch_factor,
    )
    val_dl = torch.utils.data.DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        collate_fn=custom_collate,
        drop_last=False,
    )
    # print("dummy loop")
    # for epoch in range(epochs):
    #     for batch_idx, batch in enumerate(val_dl):
    #         print('val', epoch, batch_idx, end="\r")
    #     for batch_idx, batch in enumerate(train_dl):
    #         print('train', epoch, batch_idx, end="\r")
    # exit()

    ################## NETWORK ##################
    print(gpu_number, "Setting network...")
    device = f"cuda:{gpu_number}" if torch.cuda.is_available() else "cpu"
    print(gpu_number, "Device:", device)
    archclass = EfficientViTSegL2
    net = archclass(
        use_norm_params=use_norm_params,
        pretrained=not (from_scratch or ckpt_path),
    )
    net = load_from_ckpt(net, ckpt_path)
    net.to(device)
    if os.path.exists(snapshot_path) and resume:
        loc = device
        snapshot = torch.load(snapshot_path, map_location=loc)
        net.load_state_dict(snapshot["MODEL_STATE"])
        global_step = snapshot["GLOBAL_STEP"]
        optim_state, scheduler_state = snapshot["OPTIM"], snapshot["SCHEDULER"]
        print(
            gpu_number,
            "Resuming from snapshot",
            snapshot_path,
            "at global step",
            global_step,
        )
    else:
        optim_state, scheduler_state, global_step = None, None, 0

    # transforms
    rrc = RandomResizedCrop((512,512))
    norm = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    pmd = RandomPhotometricDistort()

    ################## TRAIN INIT ##################
    print(gpu_number, "Initializing training...")
    total_steps = epochs * len(train_dl)
    val_every = (
        int(val_every_pct * total_steps) if (0 < val_every_pct < 1) else len(train_dl)
    )
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-1)

    optim = SING(net.parameters(), lr=max_lr // 10, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optim,
        max_lr=max_lr,
        total_steps=total_steps + 1,
        pct_start=0.05,
        final_div_factor=1e12,
    )
    if (optim_state is not None) and (scheduler_state is not None):
        optim.load_state_dict(optim_state)
        scheduler.load_state_dict(scheduler_state)

    ################## TRAIN LOOP ##################
    print(gpu_number, "Training...")
    epoch_init = (
        global_step // len(train_dl) if global_step > 0 else 0
    )  # resume (optional)
    batch_init = (
        global_step % len(train_dl) if global_step > 0 else 0
    )  # resume (optional)
    st = time.time()
    net.train()
    for epoch in range(epoch_init, epochs):
        logger.add_scalar("train/epoch", epoch, global_step)
        for batch_idx, batch in enumerate(train_dl):
            if batch_idx < batch_init:
                print(f"skipping {batch_idx} until batch_init...", end="\r")
                continue  # resume (optional)
            batch_init = 0
            ### AUX VARIABLES ###
            log_input_output_flag = False and (
                global_step % (total_steps // 10) == 0 or global_step == total_steps - 1
            )  # log images every 10% of steps
            validate = (
                global_step % val_every == 0 or global_step == total_steps - 1
            )  # validate every val_every_pct of total steps
            ### CORE COMPUTATIONS ###
            x, y = batch
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            xy = torch.cat([x, y], dim=1)
            if torch.rand(1) < 0.5:  # random flip
                xy = torch.flip(xy, dims=[3])
            xy = rrc(xy)
            x, y = xy[:, :3], xy[:, 3:]
            x = pmd(norm(x))
            if torch.isnan(x).any():
                print(f"nan in x at epoch {epoch} batch {batch_idx}, skipping...")
                continue
            y = y.to(torch.int64)
            y = y[:, 0]  # remove channel dim for int labels
            optim.zero_grad()
            y_hat = net(x)
            y_hat = resize(y_hat, y.shape[-2:])
            loss = loss_fn(
                y_hat,
                y,
            )
            loss.backward()
            optim.step()
            scheduler.step()

            ### LOGGING ###
            current_lr = optim.param_groups[0]["lr"]
            logger.add_scalar("train/lr", current_lr, global_step)
            logger.add_scalar("train/loss", loss.item(), global_step)

            if log_input_output_flag:
                print()
                print(gpu_number, "Saving images...")
                prediction = y_hat.argmax(dim=1, keepdim=False)
                log_input_output(
                    "train", x, prediction, global_step, img_dstdir, out_dstdir
                )

            print(
                f"GPU {gpu_number}",
                f"{(global_step / total_steps * 100):.2f}% G {global_step}/{total_steps}, "
                f"E {epoch}/{epochs}, "
                f"B {batch_idx}/{len(train_dl)}, "
                f"Loss {loss.item():.3e}, "
                # f"LR {current_lr:.3e}",
                f"Speed {(global_step + 1) / (time.time() - st):.3e} step/s",
                end="\r",
            )

            ### VALIDATION ###
            if validate:
                check_for_nan(loss, net, (x, y))
                del x, y, y_hat, loss
                gc.collect()
                # check for nan
                run_validation(
                    logger,
                    img_dstdir,
                    out_dstdir,
                    val_dl,
                    device,
                    net,
                    global_step,
                    loss_fn,
                )
                if batch_idx > 0 or epoch > 0:
                    save_checkpoint(
                        snapshot_path, logger, net, global_step, optim, scheduler
                    )
                gc.collect()

            global_step += 1

    print(gpu_number, "Training finished")
    if gpu_number == MAIN_GPU_NUMBER:
        with open(Path(logger.get_logdir()) / "done.txt", "w") as f:
            f.write("done")
        # remove snapshot
        if os.path.exists(snapshot_path):
            os.remove(snapshot_path)


def run_validation(
    logger,
    img_dstdir,
    out_dstdir,
    val_dl,
    device,
    net,
    global_step,
    loss_fn,
):
    net.eval()
    with torch.no_grad():
        iis_ious = []
        losses = []
        for batch_idx, batch in enumerate(val_dl):
            x, y = batch
            x, y = x.to(device), y.to(device)
            y = y[:, 0]  # remove channel dim for int labels
            y_hat = net(x)
            y_hat = resize(y_hat, y.shape[-2:])
            loss = loss_fn(y_hat, y)
            losses.append(loss)
            # iis_ious.append(iis_iou(y_hat, y, get_mask_from_query_fn))
            prediction = y_hat.argmax(dim=1, keepdim=False)

            # log input output for the first batch
            if batch_idx == 0:
                log_input_output(
                    "val",
                    x,
                    prediction,
                    global_step,
                    img_dstdir,
                    out_dstdir,
                )

        # average
        avg_loss = sum(losses) / len(losses)
        # avg_iou = sum(iis_ious) / len(iis_ious)
        avg_iou = -1

    # log
    logger.add_scalar(f"val/loss", avg_loss, global_step)
    # logger.add_scalar("val/iis_iou", avg_iou, global_step)

    print()
    print(
        f"Global step {global_step}, val loss {avg_loss:.3e}, val iis_iou {avg_iou:.3e}"
    )

    net.train()


def save_checkpoint(snapshot_path, logger, net, global_step, optim, scheduler):
    last_validated_model_path = Path(logger.get_logdir()) / "last_validated_model.pth"
    # snapshot
    snapshot = {
        "MODEL_STATE": net.state_dict(),
        "GLOBAL_STEP": global_step,
        "OPTIM": optim.state_dict(),
        "SCHEDULER": scheduler.state_dict(),
    }
    torch.save(snapshot, snapshot_path)
    torch.save(snapshot, last_validated_model_path)


if __name__ == "__main__":
    print("Launching `train`...")
    from fire import Fire

    Fire(train)
