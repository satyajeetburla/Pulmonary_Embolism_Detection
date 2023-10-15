import os
import cv2
import pickle
import argparse
import numpy as np
import albumentations
import pydicom
from tqdm import trange
import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
from transformers import get_linear_schedule_with_warmup
from pretrainedmodels.senet import se_resnext101_32x4d
from apex import amp


def set_seed(s):
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed(s)
    torch.backends.cudnn.deterministic = True
    random.seed(s)


def adjust_img(img_data, WL=50, WW=350):
    upper_bound, lower_bound = WL + WW // 2, WL - WW // 2
    img_data = np.clip(img_data.copy(), lower_bound, upper_bound)
    img_data = (img_data - np.min(img_data)) / (np.max(img_data) - np.min(img_data))
    return (img_data * 255).astype('uint8')


class PulmonaryDataset(Dataset):
    def __init__(self, images, bboxes, img_list, dim, augmentations):
        self.images = images
        self.bboxes = bboxes
        self.img_list = img_list
        self.dim = dim
        self.augmentations = augmentations

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_data = [pydicom.dcmread(self.images[self.img_list[idx]][key]).pixel_array for key in
                    ['image_minus1', 'image', 'image_plus1']]
        img_data = [adjust_img(d * img.RescaleSlope + img.RescaleIntercept, WL=100, WW=700) for d, img in
                    zip(img_data, self.images[self.img_list[idx]].values())]

        concat_img = np.stack(img_data, axis=-1)
        box = self.bboxes[self.images[self.img_list[idx]]['series_id']]
        cropped_img = concat_img[box[1]:box[3], box[0]:box[2], :]
        resized_img = cv2.resize(cropped_img, (self.dim, self.dim))
        transformed = self.augmentations(image=resized_img)
        x_tensor = torch.tensor(transformed['image']).permute(2, 0, 1)
        label = self.images[self.img_list[idx]]['pe_present_on_image']
        return x_tensor, label


class PulmonaryModel(nn.Module):
    def __init__(self):
        super(PulmonaryModel, self).__init__()
        self.base_model = se_resnext101_32x4d(num_classes=1000, pretrained='imagenet')
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(self.base_model.last_linear.in_features, 1)

    def forward(self, x):
        x = self.base_model.features(x)
        x = self.pooling(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


def run_training():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=-1)
    args = parser.parse_args()

    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend="nccl")

    set_seed(1001)

    # Load input data
    with open('../input_data/train_imgs.pkl', 'rb') as f:
        imgs = pickle.load(f)
    with open('../input_data/train_data.pkl', 'rb') as f:
        data = pickle.load(f)
    with open('../input_data/train_boxes.pkl', 'rb') as f:
        boxes = pickle.load(f)

    lr, bs, img_dim, epochs = 0.0004, 30, 576, 1

    model = PulmonaryModel().cuda()
    model, optimizer = amp.initialize(model, Adam(model.parameters(), lr=lr), opt_level="O1")
    model = DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)

    total_steps = len(imgs) // (bs * 4) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    loss_fn = nn.BCEWithLogitsLoss().cuda()

    train_aug = albumentations.Compose([
        albumentations.RandomContrast(limit=0.2),
        albumentations.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=20,
                                        border_mode=cv2.BORDER_CONSTANT),
        albumentations.Cutout(num_holes=2, max_h_size=int(0.4 * img_dim), max_w_size=int(0.4 * img_dim), fill_value=0),
        albumentations.Normalize(mean=[0.456] * 3, std=[0.224] * 3, max_pixel_value=255.0)
    ])

    dataset = PulmonaryDataset(imgs, boxes, list(imgs.keys()), img_dim, train_aug)
    data_loader = DataLoader(dataset, batch_size=bs, sampler=DistributedSampler(dataset), num_workers=5,
                             pin_memory=True)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in data_loader:
            images = images.cuda()
            labels = labels.float().cuda()

            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs.squeeze(), labels)
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item() * images.size(0)

        if args.local_rank == 0:
            print(f"Epoch {epoch + 1}/{epochs} - Loss: {running_loss / len(dataset)}")

            save_path = f'weights/model_epoch_{epoch + 1}.pt'
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.module.state_dict(), save_path)


if __name__ == "__main__":
    run_training()
