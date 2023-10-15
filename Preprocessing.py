import numpy as np
import pandas as pd
import pydicom
from tqdm import tqdm
import os
import glob
import pickle


def extract_dicom_data(directory):
    dcm_files = glob.glob(os.path.join(directory, '*.dcm'))
    dicom_data = [pydicom.dcmread(file) for file in dcm_files]
    z_positions = [float(data.ImagePositionPatient[-1]) for data in dicom_data]

    order = np.argsort(z_positions)
    exposures = [float(data.Exposure) for data in dicom_data]
    slice_thicknesses = [float(data.SliceThickness) for data in dicom_data]

    return np.array(dcm_files)[order], np.array(z_positions)[order], np.array(exposures)[order], \
    np.array(slice_thicknesses)[order]


data_frame = pd.read_csv('train.csv')
unique_series = data_frame['StudyInstanceUID'].astype(str) + '_' + data_frame['SeriesInstanceUID'].astype(str)
unique_series = sorted(list(set(unique_series)))

series_info = {}
image_info = {}

for idx in tqdm(range(data_frame.shape[0])):
    series_key = data_frame.loc[idx, 'StudyInstanceUID'] + '_' + data_frame.loc[idx, 'SeriesInstanceUID']
    img_key = data_frame.loc[idx, 'SOPInstanceUID']

    if series_key not in series_info:
        series_info[series_key] = {col: data_frame.loc[idx, col] for col in data_frame.columns[2:]}
        series_info[series_key]['image_order'] = []

    series_info[series_key]['image_order'].append(img_key)

    image_info[img_key] = {
        'pe_image_presence': data_frame.loc[idx, 'pe_present_on_image'],
        'parent_series': series_key,
    }

for s_key in tqdm(series_info.keys()):
    folder_path = os.path.join('../../input/train', s_key.split('_')[0], s_key.split('_')[1])
    files, z_order, exp_order, thick_order = extract_dicom_data(folder_path)

    for j, file in enumerate(files):
        img_name = os.path.basename(file).split('.')[0]
        image_info[img_name].update({
            'z_position': z_order[j],
            'exposure_level': exp_order[j],
            'slice_thickness': thick_order[j],
            'previous_img': files[j - 1].split('/')[-1].split('.')[0] if j > 0 else img_name,
            'next_img': files[j + 1].split('/')[-1].split('.')[0] if j < len(files) - 1 else img_name
        })

np.random.seed(100)
np.random.shuffle(unique_series)

img_sequence_train = [img for s in unique_series for img in series_info[s]['image_order']]

output_path = 'splitall/'
os.makedirs(output_path, exist_ok=True)

with open(os.path.join(output_path, 'series_info.pkl'), 'wb') as handle:
    pickle.dump(series_info, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(os.path.join(output_path, 'image_info.pkl'), 'wb') as handle:
    pickle.dump(image_info, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(os.path.join(output_path, 'series_order_train.pkl'), 'wb') as handle:
    pickle.dump(unique_series, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(os.path.join(output_path, 'image_order_train.pkl'), 'wb') as handle:
    pickle.dump(img_sequence_train, handle, protocol=pickle.HIGHEST_PROTOCOL)
