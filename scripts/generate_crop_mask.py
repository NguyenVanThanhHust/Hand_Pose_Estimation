import os
from os.path import isfile, isdir, join

import json
import numpy as np
import cv2

import argparse

gaussian_kernel = \
[[0.00291502 ,0.01306423 ,0.02153928 ,0.01306423 ,0.00291502] , 
[0.01306423 ,0.05854983 ,0.09653235 ,0.05854983 ,0.01306423], 
[0.02153928 ,0.09653235 ,0.15915494 ,0.09653235 ,0.02153928], 
[0.01306423 ,0.05854983 ,0.09653235 ,0.05854983 ,0.01306423],
[0.00291502 ,0.01306423 ,0.02153928 ,0.01306423 ,0.00291502]]

gaussian_kernel = np.array(gaussian_kernel)

def get_args():
    parser = argparse.ArgumentParser('Convert data', add_help=False)
    parser.add_argument('--input_folder', default="../Datasets/CMU_Hand_Dataset/hand_labels/manual_train/", type=str)
    parser.add_argument('--output_folder', default="../Datasets/CMU_Hand_Dataset/hand_labels/manual_train_processed/", type=str)
    args = parser.parse_args()
    return args 

def main(args):
    input_folder = args.input_folder
    assert isdir(input_folder), "This is not a folder: {}".format(input_folder)
    output_folder = args.output_folder
    os.makedirs(output_folder, exist_ok=True)
    im_names = next(os.walk(input_folder))[2]
    im_names = [im_name for im_name in im_names if im_name.endswith(".jpg")]
    im_names = sorted(im_names)
    im_paths = [join(input_folder, im_name) for im_name in im_names]
    label_paths = [im_path.replace(".jpg", ".json") for im_path in im_paths]
    
    num_sample = len(im_paths) // 2
    for i in range(num_sample):
        left_im_path = im_paths[2*i]
        right_im_path = im_paths[2*i+1]
        im = cv2.imread(left_im_path)
        height, width, channel = im.shape
        label = np.zeros((height+4, width+4), dtype=np.float32)
        left_json_path = label_paths[2*i]
        right_json_path = label_paths[2*i+1]
        with open(left_json_path, 'r') as handle:
            data = json.load(handle)
            hand_pts = data["hand_pts"]
            for hand_pt in hand_pts:
                if hand_pt[2] == 0.0:
                    continue
                x, y = int(hand_pt[0]), int(hand_pt[1])
                label[y-2:y+3, x-2:x+3] = gaussian_kernel
        with open(right_json_path, 'r') as handle:
            data = json.load(handle)
            hand_pts = data["hand_pts"]
            for hand_pt in hand_pts:
                if hand_pt[2] == 0.0:
                    continue
                x, y = int(hand_pt[0]), int(hand_pt[1])
                label[y-2:y+3, x-2:x+3] = gaussian_kernel
        label = label[2:-2, 2:-2]
        im_name = left_im_path.split("/")[-1]
        base_name = im_name[:-6]
        output_img_path = join(output_folder, base_name+".jpg")
        output_mask_path = join(output_folder, base_name+".npy")
        cv2.imwrite(output_img_path, im)
        np.save(output_mask_path, label)
    return 

if __name__ == "__main__":
    args = get_args()
    main(args)