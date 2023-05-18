import os
from os.path import join, isdir, isfile
import numpy as np
import cv2

import albumentations as A
from albumentations.pytorch import ToTensorV2

from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader

class Multiview_Hand_Dataset(Dataset):
    """
    """
    def __init__(self, data_folder, data_type="synthetic", split="train", transform=None):
        self.data_folder = data_folder
        self.transform = transform
        if data_type == "synthetic":
            self.im_paths = []
            self.label_paths = []
            if split=="train":
                im_folders = ["synth1", "synth2", "synth3"] 
            else:
                im_folders = ["synth4", ] 
            for im_folder in im_folders:
                folder_path = join(self.data_folder, "hand_labels_synth", im_folder)
                print(folder_path)
                im_names = next(os.walk(folder_path))[2]
                im_names = [im_name for im_name in im_names if im_name.endswith(".jpg")]
                im_paths = [join(folder_path, im_name) for im_name in im_names]
                for im_path in im_paths:
                    self.im_paths.append(im_path)
            self.label_paths = [im_path.replace(".jpg", ".json") for im_path in self.im_paths]
        else:
            if split=="train":
                self.im_folder = join(self.data_folder, "hand_labels", "manual_train")
                self.label_folder = join(self.data_folder, "hand_labels", "manual_train")
            else:
                self.im_folder = join(self.data_folder, "hand_labels", "manual_test")
                self.label_folder = join(self.data_folder, "hand_labels", "manual_test")
            im_names = next(os.walk(self.im_folder))[2]
            im_names = [im_name for im_name in im_names if im_name.endswith(".jpg")]
            label_names = [im_name.replace(".jpg", ".json") for im_name in im_names]
            self.im_paths = [join(self.im_folder, im_name) for im_name in im_names]
            self.label_paths = [join(self.im_folder, label_name) for label_name in label_names]
        self.transform = transform
    
    def __len__(self):
        return len(self.im_paths)

    def __getitem__(self, idx):
        file_name = self.im_names[idx]
        img_path = join(self.im_folder, file_name)
        mask_path = join(self.mask_folder, file_name)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        mask = self.preprocess_mask(mask)

        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]
        return image, mask
