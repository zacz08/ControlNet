import json
import cv2
import torch
import numpy as np
import os

from torch.utils.data import Dataset
import torchvision.transforms as transforms

# normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])


class MyDataset(Dataset):
    def __init__(self, data_split):

        self.data_split = data_split
        assert data_split in ['train','val','test','train_mini','val_mini'], \
            f"Invalid data_split: {self.data_split}"
        json_file = "./data/nuscenes/prompt_" + self.data_split + ".json"
        self.data = []
        with open(json_file, 'rt') as f:
            for line in f:
                self.data.append(json.loads(line))
        self.vae_scale = 8

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        source_filename = item['source']
        target_filename = item['target']
        prompt = item['prompt']

        source = torch.load('./data/nuscenes/bev_feature/' + 
                            self.data_split + '/' + source_filename)
        target = cv2.imread('./data/nuscenes/bev_seg_gt/' + 
                            self.data_split + '/' + target_filename)
        # target = cv2.resize(target, (1000, 1000))

        # Do not forget that OpenCV read images in BGR order.
        # source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        source_CHW = source.squeeze(0)
        source_HWC = source_CHW.permute(1, 2, 0)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        # Normalize source images to [0, 1].
        # TODO: Normalize input
        # source = source.astype(np.float32) / 255.0
        # source = normalize(source)

        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0

        return dict(jpg=target, txt=prompt, hint=source_HWC)

