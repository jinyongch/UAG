import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
from clearml import Logger, Task
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import fixed_seed, get_dataset, get_logger, get_model, save_checkpoint
from dataset import RSOC

parser = argparse.ArgumentParser(description="Train")
parser.add_argument("--net", default="UDG", type=str)
parser.add_argument("--gpu", default="0", type=str)
parser.add_argument("--dataset_root", default="/tmp/rsoc", type=str)
parser.add_argument("--dataset", default="building", type=str)
parser.add_argument("--results_root", default="./results", type=str)
parser.add_argument("--task", default="1", type=str)
parser.add_argument("--epochs", default=400, type=int)
parser.add_argument("--batch_size", default=32, type=int)
parser.add_argument("--drop_stages", nargs="+", default=[0], type=int)
parser.add_argument("--ptb_stages", nargs="+", default=[0], type=int)
parser.add_argument("--fixed", action="store_true", default=False)
parser.add_argument("--resume", default="", type=str)
parser.add_argument("--downsample", action="store_false", default=True)


def main():
    global args, best_mae, best_mse

    best_mae = 1e6
    best_mse = 1e6

    args = parser.parse_args()

    args.original_lr = 1e-7
    args.lr = 1e-7
    args.momentum = 0.95
    args.decay = 5 * 1e-4
    args.start_epoch = 0
    args.steps = [-1, 1, 100, 150]
    args.scales = [1, 1, 1, 1]
    args.workers = 8
    args.print_freq = 30

    drop_id_dir = "D_"
    for id in args.drop_stages:
        drop_id_dir += str(id) + "_"
    ptb_id_dir = "P_"
    for id in args.ptb_stages:
        ptb_id_dir += str(id) + "_"

    Task.init(
        project_name="Counting@RS",
        task_name=args.net
        + "_"
        + drop_id_dir
        + ptb_id_dir
        + args.dataset
        + "_"
        + args.task,
    )

    results_root = args.results_root
    output_dir = f"{results_root}/{args.net}_{drop_id_dir}{ptb_id_dir}{args.dataset}"
    task_dir = f"{output_dir}/{args.task}"


    writer.add_scalar("loss", losses.avg, epoch)
    Logger.current_logger().report_scalar("loss", "epoch", losses.avg, epoch)


def validate(val_list, model, logger, epoch):
    logger.info("begin val ...")
    test_loader = DataLoader(
        val_list, shuffle=False, batch_size=1, num_workers=args.workers, drop_last=False
    )

    model.eval()
    mae = 0
    mse = 0

    for it, (img, target) in enumerate(tqdm(test_loader, desc="Val", leave=False)):
        with torch.no_grad():
            img = img.cuda()
            output = model(img)

            target = target.type(torch.FloatTensor).cuda()

            est = output.sum().item()
            gt = target.sum().item()

            mae += abs(est - gt)
            mse += (est - gt) * (est - gt)

            if (it + 1) % 5 == 0:
                logger.info(f"[{it+1}] [gt: {gt:.2f}] [est: {est:.2f}]")

    mae = mae / len(test_loader)
    mse = np.sqrt(mse / len(test_loader))

    logger.info(f"[epoch: {epoch}][MAE: {mae:.3f}] [MSE: {mse:.3f}]")

    return mae, mse


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""

    args.lr = args.original_lr

    for i in range(len(args.steps)):
        scale = args.scales[i] if i < len(args.scales) else 1

        if epoch >= args.steps[i]:
            args.lr = args.lr * scale
            if epoch == args.steps[i]:
                break
        else:
            break
    for param_group in optimizer.param_groups:
        param_group["lr"] = args.lr


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == "__main__":
    main()

    for directory in [results_root, output_dir, task_dir]:
        os.makedirs(directory, exist_ok=True)

    logger = get_logger("Train", f"{output_dir}/{args.task}_train.log", "w")
    writer = SummaryWriter(task_dir)

    if args.fixed:
        fixed_seed(821)
    else:
        fixed_seed(time.time())

    train_list, val_list = get_dataset(args.dataset, args.dataset_root)

    train_list = RSOC(train_list, True, args.downsample)
    val_list = RSOC(val_list, False, args.downsample)

    model = get_model(args)

    criterion = nn.MSELoss(size_average=False).cuda()

    optimizer = torch.optim.SGD(
        model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.decay
    )

    if args.resume:
        if os.path.isfile(args.resume):
            logger.info("=> loading checkpoint '{}'".format(args.resume))

            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint["epoch"]
            best_mae = checkpoint["best_mae"]
            model.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer"])

            logger.info(
                "=> loaded checkpoint '{}' (epoch {})".format(
                    args.resume, checkpoint["epoch"]
                )
            )
        else:
            logger.info("=> no checkpoint found at '{}'".format(args.resume))

    logger.info(f"[net: {args.net}]")
    logger.info(f"[gpu: {args.gpu}]")
    logger.info(f"[dataset_root: {args.dataset_root}]")
    logger.info(f"[dataset: {args.dataset}]")
    logger.info(f"[task: {args.task}]")
    logger.info(f"[max_epoch: {args.epochs}]")
    logger.info(f"[batch_size: {args.batch_size}]")
    logger.info(f"[drop_stages: {args.drop_stages}]")
    logger.info(f"[ptb_stages: {args.ptb_stages}]")
    logger.info(f"[fixed: {args.fixed}]")
    logger.info(f"[downsample: {args.downsample}]")
    logger.info(f"[resume: {args.resume}]")

    for epoch in tqdm(
        range(args.start_epoch, args.epochs),
        desc=f"{args.net}_{args.dataset}_{args.task}",
        leave=False,
    ):
        adjust_learning_rate(optimizer, epoch)

        train(train_list, model, criterion, optimizer, epoch + 1, logger, writer)
        mae, mse = validate(val_list, model, logger, epoch + 1)

        for key, value in {"mae": mae, "mse": mse}.items():
            writer.add_scalar(key, value, epoch + 1)
            Logger.current_logger().report_scalar(key, "epoch", value, epoch + 1)

        is_best = mae < best_mae
        if is_best:
            best_mae = mae
            best_mse = mse

        logger.info(" best MAE {mae:.3f} ".format(mae=best_mae))
        logger.info(" best MSE {mse:.3f} ".format(mse=best_mse))

        save_checkpoint(
            {
                "epoch": epoch,
                "resume": args.resume,
                "state_dict": model.state_dict(),
                "best_mae": best_mae,
                "optimizer": optimizer.state_dict(),
            },
            is_best,
            args.task,
            output_dir,
        )


def train(train_list, model, criterion, optimizer, epoch, logger, writer):
    losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    train_loader = DataLoader(
        train_list,
        shuffle=True,
        batch_size=args.batch_size,
        num_workers=args.workers,
        drop_last=True,
    )

    logger.info(
        "epoch %d, processed %d samples, lr %.10f"
        % (epoch, len(train_loader.dataset), args.lr)
    )

    model.train()
    end = time.time()

    for it, (img, target) in enumerate(tqdm(train_loader, desc="Train", leave=False)):
        data_time.update(time.time() - end)

        img = img.cuda()
        output = model(img)

        target = target.type(torch.FloatTensor).unsqueeze(1).cuda()

        loss = criterion(output, target)

        losses.update(loss.item(), img.size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if (it + 1) % args.print_freq == 0:
            logger.info(
                "Epoch: [{0}][{1}/{2}]\t"
                "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                "Data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                "Loss {loss.val:.4f} ({loss.avg:.4f})\t".format(
                    epoch,
                    it + 1,
                    len(train_loader),
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=losses,
                )
            )