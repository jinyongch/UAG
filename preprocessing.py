import glob
import os

import h5py
import numpy as np
import scipy
import scipy.io as io
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter


# this is borrowed from https://github.com/davideverona/deep-crowd-counting_crowdnet
def gaussian_filter_density(gt):
    density = np.zeros(gt.shape, dtype=np.float32)
    gt_count = np.count_nonzero(gt)
    if gt_count == 0:
        return density

    # pts = np.array(zip(np.nonzero(gt)[1], np.nonzero(gt)[0]))
    pts = np.array(list(zip(np.nonzero(gt)[1].ravel(), np.nonzero(gt)[0].ravel())))
    leafsize = 2048
    # build kdtree
    tree = scipy.spatial.KDTree(pts.copy(), leafsize=leafsize)
    # query kdtree
    distances, locations = tree.query(pts, k=4)

    print("generate density...")
    for i, pt in enumerate(pts):
        pt2d = np.zeros(gt.shape, dtype=np.float32)
        pt2d[pt[1], pt[0]] = 1.0
        if gt_count > 1:
            sigma = (distances[i][1] + distances[i][2] + distances[i][3]) * 0.1
        else:
            sigma = np.average(np.array(gt.shape)) // 2.0 // 2.0  # case: 1 point
        density += gaussian_filter(pt2d, sigma, mode="constant")
    print("done.")
    return density


# set the root to the building dataset you download
# root = "/scratch/rsoc/building"
root = "/scratch/rsoc/part_B"

# now generate the RSOC_building ground truth
building_train = os.path.join(root, "train_data", "images")
building_test = os.path.join(root, "test_data", "images")

path_sets = [building_train, building_test]

img_paths = []
for path in path_sets:
    for img_path in glob.glob(os.path.join(path, "*.jpg")):
        img_paths.append(img_path)

for img_path in img_paths:
    print(img_path)
    mat = io.loadmat(
        img_path.replace(".jpg", ".mat")
        .replace("images", "ground_truth")
        .replace("IMG_", "GT_IMG_")
    )
    img = plt.imread(img_path)
    k = np.zeros((img.shape[0], img.shape[1]))
    # gt = mat["center"][0, 0]
    gt = mat["image_info"][0, 0][0, 0][0]
    for i in range(0, len(gt)):
        if int(gt[i][1]) < img.shape[0] and int(gt[i][0]) < img.shape[1]:
            k[int(gt[i][1]), int(gt[i][0])] = 1

    k = gaussian_filter_density(k)
    # k = gaussian_filter(k, 15)
    groundtruth = np.asarray(k)

    # groundtruth = np.array([groundtruth])
    groundtruth = groundtruth[None, :, :]

    with h5py.File(
        img_path.replace(".jpg", ".h5").replace("images", "ground_truth_jpg"), "w"
    ) as hf:
        hf["density_map"] = groundtruth
