import argparse
import os

import h5py
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from config import fixed_seed, get_dataset, get_logger, get_model, save_density_map


def enable_dropout(model):
    """Function to enable the dropout layers during test-time"""
    for m in model.modules():
        if m.__class__.__name__.startswith("Dropout"):
            m.train()


transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

parser = argparse.ArgumentParser(description="Test")
parser.add_argument("--net", default="UDG", type=str)
parser.add_argument("--gpu", default="0", type=str)
parser.add_argument("--dataset_root", default="/tmp/rsoc", type=str)
parser.add_argument("--dataset", default="building", type=str)
parser.add_argument("--results_root", default="./results", type=str)
parser.add_argument("--task", default="1", type=str)
parser.add_argument("--drop_stages", nargs="+", default=[0], type=int)
parser.add_argument("--ptb_stages", nargs="+", default=[0], type=int)
parser.add_argument(
    "--mc_drop",
    action="store_true",
    default=False,
    help="Use MC Dropout to consider model uncertainty.",
)
parser.add_argument(
    "--n_forward",
    default=0,
    type=int,
    help="Number of forward passes. Effective only when MC Dropout is used",
)
parser.add_argument("--model_best", default="", type=str)
parser.add_argument("--fixed", action="store_true", default=False)

args = parser.parse_args()

drop_id_dir = "D_"
for id in args.drop_stages:
    drop_id_dir += str(id) + "_"
ptb_id_dir = "P_"
for id in args.ptb_stages:
    ptb_id_dir += str(id) + "_"

output_dir = f"{args.results_root}/{args.net}_{drop_id_dir}{ptb_id_dir}{args.dataset}"
task_dir = f"{output_dir}/{args.task}_{args.n_forward}"
gt_dir = f"{task_dir}/gt"
est_dir = f"{task_dir}/est"

# Create the directories if they don't exist
for directory in [task_dir, gt_dir, est_dir]:
    os.makedirs(directory, exist_ok=True)

logger = get_logger("Test", f"{output_dir}/{args.task}_{args.n_forward}_test.log", "w")

logger.info(f"[net: {args.net}]")
logger.info(f"[gpu: {args.gpu}]")
logger.info(f"[dataset_root: {args.dataset_root}]")
logger.info(f"[dataset: {args.dataset}]")
logger.info(f"[results_root: {args.results_root}]")
logger.info(f"[task: {args.task}]")
logger.info(f"[drop_stages: {args.drop_stages}]")
logger.info(f"[ptb_stages: {args.ptb_stages}]")
logger.info(f"[mc_drop: {args.mc_drop}]")
logger.info(f"[n_forward: {args.n_forward}]")
logger.info(f"[model_best: {args.model_best}]")
logger.info(f"[fixed: {args.fixed}]")

if args.fixed:
    fixed_seed(seed=821)

logger.info("load model ...")
model = get_model(args)

logger.info("load model parameters ...")
if args.model_best:
    model_best_path = args.model_best
else:
    model_best_path = f"{output_dir}/{args.task}_model_best.pth.tar"
    # model_best_path = f"{output_dir}/{args.task}model_best.pth.tar"
checkpoint = torch.load(model_best_path)
model.load_state_dict(checkpoint["state_dict"])

logger.info("load dataset ...")
test_list = get_dataset(args.dataset, args.dataset_root, False)


logger.info("begin test ...")
model.eval()
mae = 0
mse = 0


for img_path in tqdm(
    test_list,
    desc=f"{args.net}_{drop_id_dir}{ptb_id_dir}{args.dataset}_{args.task}_{args.n_forward}",
    leave=False,
):
    file_path, filename = os.path.split(img_path)

    img = Image.open(img_path).convert("RGB")
    img = transform(img)

    gt_path = (
        img_path.replace(".png", ".h5")
        .replace(".jpg", ".h5")
        .replace("images_bk", "ground_truth")
        .replace("images", "ground_truth")
    )

    with h5py.File(gt_path, "r") as f:
        target = np.asarray(f["density_map"])
        target = target.squeeze()

    with torch.no_grad():
        img = img.unsqueeze(0).cuda()

        if args.mc_drop:
            outputs = []
            enable_dropout(model)
            for i in range(args.n_forward):
                outputs.append(model(img))
            output = torch.stack(outputs).mean(dim=0)
        else:
            output = model(img)

        output = output.squeeze().detach().cpu().numpy()

        est = output.sum()
        gt = target.sum()

        logger.info(f"[{filename}] [gt: {int(gt):.2f}] [est: {est:.2f}]")

        save_density_map(target, f"{gt_dir}/{filename}")
        save_density_map(output, f"{est_dir}/{filename}")

        mae += abs(est - gt)
        mse += (est - gt) * (est - gt)

mae = mae / len(test_list)
mse = np.sqrt(mse / len(test_list))

logger.info(f"{drop_id_dir}{ptb_id_dir}mae_{mae:.2f}_rmse_{mse:.2f}")
