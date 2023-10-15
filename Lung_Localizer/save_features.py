import numpy as np
import pandas as pd
import os
import cv2
from tqdm import tqdm
import torch.nn as nn
import torch
from efficientnet_pytorch import EfficientNet
import pydicom
import glob

def window_image(img_data, center, width):
    img_min, img_max = center - width // 2, center + width // 2
    img_data = np.clip(img_data, img_min, img_max)
    img_data = (img_data - img_min) / (img_max - img_min)
    return img_data

class MedicalImageDataset(torch.utils.data.Dataset):
    def __init__(self, img_series):
        self.img_series = img_series
    def __len__(self):
        return len(self.img_series)
    def __getitem__(self, idx):
        return idx

class DICOMDataCollator:
    def __init__(self, img_series):
        self.img_series = img_series

    def fetch_dicom_data(self, series_folder):
        dicom_files = sorted(glob.glob(os.path.join(series_folder, '*.dcm')), key=lambda x: pydicom.dcmread(x).ImagePositionPatient[-1])
        dicoms = [pydicom.dcmread(file) for file in dicom_files]
        slope = float(dicoms[0].RescaleSlope)
        intercept = float(dicoms[0].RescaleIntercept)
        selection_indices = [int(ratio*len(dicom_files)) for ratio in [0.2, 0.3, 0.4, 0.5]]
        selected_files = [dicom_files[idx] for idx in selection_indices]
        selected_dicoms = [dicoms[idx] for idx in selection_indices]
        pixel_data = np.array([img.pixel_array.astype(np.float32) for img in selected_dicoms])
        pixel_data = pixel_data * slope + intercept
        return window_image(pixel_data, 100, 700), dicom_files, selected_files

    def __call__(self, idx_batch):
        series_info = self.img_series[idx_batch[0]]
        study_info, series_id = series_info.split('_')
        series_path = os.path.join('../../../input/train/', study_info, series_id)
        processed_imgs, all_files, selected_files = self.fetch_dicom_data(series_path)
        images_tensor = torch.from_numpy(processed_imgs).unsqueeze(1).repeat(1, 3, 1, 1)
        return images_tensor, [file[-16:-4] for file in all_files], [file[-16:-4] for file in selected_files], series_info

class LungBoxNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_net = EfficientNet.from_pretrained('efficientnet-b0')
        features_count = self.base_net._fc.in_features
        self.output_layer = nn.Linear(features_count, 4)
    def forward(self, x):
        x = self.base_net.extract_features(x)
        x = self.base_net._avg_pooling(x)
        x = x.view(x.size(0), -1)
        return self.output_layer(x)

def run_pipeline():
    with open('../../process_input/splitall/series_list_train.pickle', 'rb') as f:
        img_series = pickle.load(f)
    bbox_data = pd.read_csv('../lung_bbox.csv')
    bbox_annotations = {img_id: [max(0.0, x_min), max(0.0, y_min), min(1.0, x_max), min(1.0, y_max)]
                        for img_id, x_min, y_min, x_max, y_max in zip(bbox_data['Image'].values,
                                                                     bbox_data['Xmin'].values,
                                                                     bbox_data['Ymin'].values,
                                                                     bbox_data['Xmax'].values,
                                                                     bbox_data['Ymax'].values)}

    model = LungBoxNet()
    model.load_state_dict(torch.load('weights/epoch34_polyak'))
    model = model.cuda()
    model.eval()

    predicted_boxes = np.zeros((len(img_series)*4, 4), dtype=np.float32)
    train_annotations = {}
    selected_img_series = []

    data_feed = MedicalImageDataset(img_series=img_series)
    data_collator = DICOMDataCollator(img_series=img_series)
    data_loader = torch.utils.data.DataLoader(dataset=data_feed, collate_fn=data_collator, batch_size=1, shuffle=False, num_workers=20, pin_memory=True)

    for step, (img_data, all_img_ids, selected_img_ids, series) in tqdm(enumerate(data_loader), total=len(data_loader)):
        with torch.no_grad():
            start_idx = step*4
            end_idx = start_idx+4 if step != len(data_loader) - 1 else len(data_loader.dataset)*4
            img_data = img_data.cuda()
            predictions = model(img_data)
            boxes = predictions.cpu().data.numpy()
            predicted_boxes[start_idx:end_idx] = boxes
            selected_img_series.extend(selected_img_ids)
            final_box = [int(coord * 512) for coord in [np.min(boxes[:, i]) for i in range(4)]]
            train_annotations[series] = final_box

    loss_val = sum([abs(predicted_boxes[i, j] - bbox_annotations[selected_img_series[i]][j]) for i in range(len(img_series)*4) for j in range(4)])
    print("Total loss:", loss_val / (len(img_series) * 4))

    with open('train_annotations.pickle', 'wb') as output_file:
        pickle.dump(train_annotations, output_file, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    run_pipeline()
