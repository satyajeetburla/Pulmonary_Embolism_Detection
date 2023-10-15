import os
import cv2
import numpy as np
import pickle
import pydicom
import torch
import pandas as pd
from sklearn.metrics import roc_auc_score
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torch import nn, optim
from tqdm.notebook import tqdm
from pretrainedmodels.senet import se_resnext101_32x4d
from apex import amp


class StatTracker:
    def __init__(self):
        self.reset_stats()

    def reset_stats(self):
        self.current = 0
        self.total = 0
        self.num = 0
        self.mean = 0

    def record(self, value, size=1):
        self.current = value
        self.total += value * size
        self.num += size
        self.mean = self.total / self.num


def adjust_image_brightness(img, center=50, width=350):
    top, bottom = center + width // 2, center - width // 2
    adjusted = np.clip(img, bottom, top)
    adjusted -= adjusted.min()
    adjusted /= adjusted.max()
    return (adjusted * 255).astype('uint8')


class PEDataLoader(Dataset):
    def __init__(self, img_data, bounding_box, img_keys, dim):
        self.img_data = img_data
        self.bounding_box = bounding_box
        self.img_keys = img_keys
        self.dim = dim

    def __len__(self):
        return len(self.img_keys)

    def fetch_image(self, path):
        data = pydicom.dcmread(path)
        image = data.pixel_array * data.RescaleSlope + data.RescaleIntercept
        return adjust_image_brightness(image, 100, 700)

    def __getitem__(self, idx):
        img_key = self.img_keys[idx]
        path_data = self.img_data[img_key]
        series = path_data['series_id'].split('_')

        prev_path = f'../../input/train/{series[0]}/{series[1]}/{path_data["image_minus1"]}.dcm'
        curr_path = f'../../input/train/{series[0]}/{series[1]}/{img_key}.dcm'
        next_path = f'../../input/train/{series[0]}/{series[1]}/{path_data["image_plus1"]}.dcm'

        img_stack = [self.fetch_image(path) for path in [prev_path, curr_path, next_path]]
        img_concat = np.stack(img_stack, axis=-1)
        bbox = self.bounding_box[path_data['series_id']]
        cropped = img_concat[bbox[1]:bbox[3], bbox[0]:bbox[2], :]
        final_img = cv2.resize(cropped, (self.dim, self.dim))
        tensor_img = transforms.ToTensor()(final_img)
        normed_img = transforms.Normalize(mean=[0.456, 0.456, 0.456], std=[0.224, 0.224, 0.224])(tensor_img)

        label = path_data['pe_present_on_image']
        return normed_img, label


class CustomSEResNeXt(nn.Module):
    def __init__(self):
        super().__init__()
        self.core_model = se_resnext101_32x4d(pretrained='imagenet')
        self.pool_layer = nn.AdaptiveAvgPool2d(1)
        self.final_layer = nn.Linear(self.core_model.last_linear.in_features, 1)

    def forward(self, input_data):
        x = self.core_model.features(input_data)
        x = self.pool_layer(x)
        flattened = x.view(x.size(0), -1)
        output = self.final_layer(flattened)
        return flattened, output


def evaluate():
    with open('../process_input/splitall/image_list_train.pickle', 'rb') as handle:
        train_images = pickle.load(handle)
    with open('../process_input/splitall/image_dict.pickle', 'rb') as handle:
        image_info = pickle.load(handle)
    with open('../lung_localization/splitall/bbox_dict_train.pickle', 'rb') as handle:
        bbox_info = pickle.load(handle)

    model = CustomSEResNeXt()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    loss_fn = nn.BCEWithLogitsLoss().cuda()

    checkpoints = ['epoch0']
    for checkpoint in checkpoints:
        model.load_state_dict(torch.load(f'weights/{checkpoint}'))
        model = model.cuda()
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1", verbosity=0)
        model = torch.nn.DataParallel(model) if torch.cuda.device_count() > 1 else model
        model.eval()

        data = PEDataLoader(image_info, bbox_info, train_images, 576)
        data_loader = DataLoader(data, batch_size=96, shuffle=False, num_workers=18, pin_memory=True)

        loss_tracker = StatTracker()
        features, predictions = [], []

        for img, label in tqdm(data_loader):
            img, label = img.cuda(), label.float().cuda()
            with torch.no_grad():
                feature_out, pred_out = model(img)
                loss = loss_fn(pred_out.view(-1), label)
                loss_tracker.record(loss.item(), img.size(0))
                features.append(feature_out.cpu().numpy())
                predictions.append(pred_out.sigmoid().cpu().numpy())

        final_features = np.concatenate(features)
        final_predictions = np.concatenate(predictions)

        os.makedirs('features0/', exist_ok=True)
        np.save('features0/feature_train', final_features)
        np.save('features0/pred_prob_train', final_predictions)

        actual_labels = [image_info[img_id]['pe_present_on_image'] for img_id in train_images]
        auc_score = roc_auc_score(actual_labels, final_predictions)

        print(f"Checkpoint: {checkpoint} | Loss: {loss_tracker.mean:.4f} | AUC: {auc_score:.4f}")


if __name__ == "__main__":
    evaluate()
