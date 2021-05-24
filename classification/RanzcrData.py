import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

'''
Author: Kayed Mahra
Returns a Dataset object to be used with dataloader. Images and binary masks are read as 3 identical channels.
The last channel of the image is subsituted for the binary mask. The sample consisting of the two is then normalized [0,255]
'''
class RanzcrDatasetClassification(Dataset):
    def __init__(self, labels, img_dir, mask_dir, ids, shuffle=True):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.labels = labels
        self.id_set = ids

    def __len__(self):
        return len(self.id_set)
    
    def __getitem__(self, idx):
        sample_id = self.id_set[idx]
        img_path = os.path.join(self.img_dir, sample_id + '.jpg')
        mask_path = os.path.join(self.mask_dir, sample_id + '.png')
        img, mask = cv2.imread(img_path).astype(np.float32), cv2.imread(mask_path).astype(np.float32)
        img[:,:,2] = mask[:,:,0]

        image = img.transpose(2, 0, 1) / 255.
        label = np.array(self.labels.query('StudyInstanceUID' + f'== "{sample_id}"').values[0,1:-1], dtype=np.float32) # Extract the 11 targets
        
        return torch.tensor(image).float(), torch.tensor(label).float()