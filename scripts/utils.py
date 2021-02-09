import re
import cv2
import numpy as np
from tqdm import tqdm
from pathlib import Path
from .image import RGBDImage


def extract_rgbd_result(rgb_dir_path, depth_dir_path, max_depth=100.0):
    rgbd_obj_list = []
    rgb_path_list = [path for path in rgb_dir_path.glob("*") if path.suffix in [".png", ".jpg", ".jpeg"]]

    print("Data is loading...")
    for rgb_image_path in tqdm(rgb_path_list):
        rgb_image_name = rgb_image_path.name
        rgb_image_suffix = rgb_image_path.suffix
        image_height, image_width, _ = cv2.imread(str(rgb_image_path)).shape
        depth_image_path = Path(depth_dir_path, rgb_image_name.replace(rgb_image_suffix, ".png"))

        # print([key for key in depth_npz.keys()])  -> 'depth', 'plane', 'score'
        rgbd_obj = RGBDImage(rgb_image_path, depth_image_path)
        rgbd_obj_list.append(rgbd_obj)
    return rgbd_obj_list
