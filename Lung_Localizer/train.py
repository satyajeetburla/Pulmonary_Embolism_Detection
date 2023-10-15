import argparse
import os
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm
import torch
from torch import nn, optim
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from efficientnet_pytorch import EfficientNet
import random
import pickle
import albumentations as A
import pydicom
from apex import amp

class MetricsTracker:
    def __init__(self):
        self.clear()
    def clear(self):
        self.current = 0
        self.total = 0
        self.num_entries = 0
        self.mean = 0
    def record(self, value, qty=1):
        self.current = value
        self.total += value * qty
        self.num_entries += qty
        self.mean = self.total / self.num_entries

def image_windowing(img_data, level=50, width=350):
    upper_bound, lower_bound = level + width // 2, level - width // 2
    img_data = np.clip(img_data, lower_bound, upper_bound)
    img_data = (img_data - img_data.min()) / (img_data.max() - img_data.min())
    return img_data

class PulmonaryEmbDataset(Dataset):
    def __init__(self, img_meta, bbox_meta, img_keys, resize_dim, augments):
        self.img_meta = img_meta
        self.bbox_meta = bbox_meta
        self.img_keys = img_keys
        self.resize_dim = resize_dim
        self.augments = augments
    def __len__(self):
        return len(self.img_keys)
    def __getitem__(self, idx):
        key = self.img_keys[idx]
        study, series = self.img_meta[key]['series_id'].split('_')
        dicom_data = pydicom.dcmread(f'../../../input/train/{study}/{series}/{key}.dcm')
        img_array = dicom_data.pixel_array.astype(np.float32)
        img_array = img_array * dicom_data.RescaleSlope + dicom_data.RescaleIntercept
        img_array = image_windowing(img_array, 100, 700)
        img_rgb = np.repeat(img_array[..., np.newaxis], 3, axis=2)
        img_rgb = cv2.resize(img_rgb, (self.resize_dim, self.resize_dim))
        aug_result = self.augments(image=img_rgb, bboxes=[self.bbox_meta[key]], class_labels=['lung'])
        img_rgb = aug_result['image'].transpose(2, 0, 1)
        bbox = torch.tensor(aug_result['bboxes'][0])
        return img_rgb, bbox

class LungModel(nn.Module):
    def __init__(self):
        super(LungModel, self).__init__()
        self.core_net = EfficientNet.from_pretrained('efficientnet-b0')
        in_feats = self.core_net._fc.in_features
        self.final_layer = nn.Linear(in_feats, 4)
    def forward(self, x):
        x = self.core_net.extract_features(x)
        x = self.core_net._avg_pooling(x)
        x = x.view(x.size(0), -1)
        return self.final_layer(x)

def train_routine():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=-1)
    args = parser.parse_args()
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend="nccl")
    device = torch.device("cuda", args.local_rank)
    args.device = device
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    train_data = {
        'series_list': pickle.load(open('../../process_input/splitall/series_list_train.pickle', 'rb')),
        'series_meta': pickle.load(open('../../process_input/splitall/series_dict.pickle', 'rb')),
        'img_meta': pickle.load(open('../../process_input/splitall/image_dict.pickle', 'rb')),
        'bbox_data': pd.read_csv('../lung_bbox.csv')
    }
    train_data['bbox_meta'] = {row['Image']: [max(0.0, row['Xmin']), max(0.0, row['Ymin']), min(1.0, row['Xmax']), min(1.0, row['Ymax'])] for _, row in train_data['bbox_data'].iterrows()}
    train_keys = []
    for series in train_data['series_list']:
        imgs = train_data['series_meta'][series]['sorted_image_list']
        select_indices = [int(factor * len(imgs)) for factor in [0.2, 0.3, 0.4, 0.5]]
        train_keys.extend([imgs[i] for i in select_indices])

    params = {
        'learning_rate': 4e-4,
        'batch_size': 32,
        'resize': 512,
        'averaging_count': 32,
        'epochs': 1000
    }

    if args.local_rank != 0:
        torch.distributed.barrier()

    net = LungModel().to(args.device)

    if args.local_rank == 0:
        torch.distributed.barrier()

    optimz = optim.Adam(net.parameters(), lr=params['learning_rate'])
    net, optimz = amp.initialize(net, optimz, opt_level="O1", verbosity=0)
    net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[args.local_rank])

    train_augments = A.Compose([
        A.Cutout(num_holes=8, max_h_size=16, max_w_size=16, always_apply=True),
        A.BboxParams(format='pascal_voc', label_fields=['class_labels'])
    ])

    train_set = PulmonaryEmbDataset(train_data['img_meta'], train_data['bbox_meta'], train_keys, params['resize'], train_augments)
    train_sampler = DistributedSampler(train_set, shuffle=True)
    train_loader = DataLoader(train_set, batch_size=params['batch_size'], sampler=train_sampler, num_workers=4)

    criterion = nn.L1Loss()
    loss_monitor = MetricsTracker()

    for epoch in range(params['epochs']):
        net.train()
        loss_monitor.clear()
        for imgs, bboxes in tqdm(train_loader):
            imgs, bboxes = imgs.to(args.device), bboxes.to(args.device)
            optimz.zero_grad()
            preds = net(imgs.float())
            loss_value = criterion(preds, bboxes)
            with amp.scale_loss(loss_value, optimz) as scaled_loss:
                scaled_loss.backward()
            optimz.step()
            loss_monitor.record(loss_value.item())
            if args.local_rank == 0:
                print(f"Epoch: {epoch}, Loss: {loss_monitor.current:.4f}, Avg. Loss: {loss_monitor.mean:.4f}")

    if args.local_rank == 0:
        save_path = f"./lung_model_epoch_{epoch}.pth"
        torch.save(net.state_dict(), save_path)

if __name__ == "__main__":
    train_routine()
