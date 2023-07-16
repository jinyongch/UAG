import importlib
import json
import logging
import os
import random
import shutil

import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm


class TqdmLoggingHandler(logging.Handler):
    def emit(self, record):
        msg = self.format(record)
        # output to terminal
        tqdm.write(msg)


def get_logger(name="Train", save_path="./results/train.log", mode="w"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    tqdm_handler = TqdmLoggingHandler()
    file_handler = logging.FileHandler(save_path, mode)

    logger.addHandler(tqdm_handler)
    logger.addHandler(file_handler)

    logger.info("-" * 25 + f" {name} " + "-" * 25)

    return logger


def fixed_seed(seed=821):
    random.seed(seed)
    np.random.seed(int(seed))
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_model(args):
    # dynamic import module
    module = importlib.import_module("models")
    # string -> class_name
    net = getattr(module, args.net)

    model = net(drop_stages_id=args.drop_stages, ptb_stages_id=args.ptb_stages)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    model = model.cuda()

    return model


def save_checkpoint(state, is_best, task_id, output_dir):
    pth_path = f"{output_dir}/{task_id}_checkpoint.pth.tar"
    best_pth_path = f"{output_dir}/{task_id}_model_best.pth.tar"

    torch.save(state, pth_path)
    if is_best:
        shutil.copyfile(pth_path, best_pth_path)


def get_dataset(dataset="ship", root_dir="/tmp/rsoc", train=True):
    train_json = f"./datasets/{dataset}_train.json"
    val_json = f"./datasets/{dataset}_val.json"
    test_json = f"./datasets/{dataset}_test.json"

    with open(train_json, "r") as f:
        train_list = json.load(f)
        revise_path(train_list, root_dir)
    with open(val_json, "r") as f:
        val_list = json.load(f)
        revise_path(val_list, root_dir)
    with open(test_json, "r") as f:
        test_list = json.load(f)
        revise_path(test_list, root_dir)

    if train:
        return train_list, val_list
    else:
        return test_list


def revise_path(path_list, root_dir="/tmp/rsoc"):
    for i in range(len(path_list)):
        path_list[i] = os.path.join(root_dir, path_list[i])


def save_density_map(density_map, save_path):
    density_map[density_map < 0] = 0
    density_map = 255 * density_map / np.max(density_map)
    density_map = density_map.astype(np.uint8)
    density_map = cv2.applyColorMap(density_map, 2)
    cv2.imwrite(save_path, density_map)


def np2img(img_np, save_path):
    height, width = img_np.shape

    figure, axes = plt.subplots()
    axes.imshow(img_np, cmap="jet")
    plt.axis("off")

    figure.set_size_inches(width / 300, height / 300)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    plt.margins(0, 0)

    plt.savefig(save_path, dpi=300)
    plt.close()


def str2conf(input):
    configs = input.split(" ")
    for config in configs:
        print(f'"{config}",')
