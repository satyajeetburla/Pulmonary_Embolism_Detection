# Pulmonary_Embolism_Detection
Find data from the below link : 
https://www.kaggle.com/competitions/rsna-str-pulmonary-embolism-detection/data
## How to use the Code:
1. First Preprocess the input
```
Python Preprocessing.py
```
2. Lung Localizer Step

```
python train.py
```
```
python save_features.py
```
3. Main Training Process
```
python train.py
```
```
python save_features.py
```

## Approach

1. Image Preparation:
It was observed that enhancing the dimensions of an image significantly boosts the accuracy of analysis. However, within the provided images, the lung representation varied both in terms of size and position.
To rectify this, an algorithm was designed to pinpoint the lungs within these images. The technique employed an Efficientnet-b0 model, and the localization was refined using specific bounding boxes for the lung areas.
For efficiency, a subset of images (four per study) was processed and trained upon.

2. Dataset Division for Training:
Given the extensive nature and high resolution of the dataset, a single partition was deemed sufficient. Specifically, a set of 1,000 studies was kept for validating model performance, leaving over 6,200 studies for model training and refinement.
For the final performance evaluation, all training data was utilized.
Detailed Image Analysis:

3. The adopted approach mirrored a two-phase training method seen in past competitions:
Image analysis was bolstered by incorporating the data from adjacent images. This multi-image technique proved to be more efficient than standalone methods.
Notably, the model was calibrated using only direct image data, even when more extensive study-level data was available.
Training incorporated binary cross-entropy loss, and image alterations (augmentations) were applied to enhance model robustness. The most effective model structures were identified as seresnext50 and seresnext101.
Broad Study Analysis:

4. For a more holistic approach, image representations (or embeddings) of size 2048 were channeled into a recurrent neural structure:
Given the variable image counts across studies, an adaptive embedding resizing technique was introduced. This ensured uniformity without altering the raw images.
The final neural design utilized a bidirectional GRU layout, with study predictions derived from a combination of attentive and max pooling mechanisms.
Refinement & Consistency:

5. The final step emphasized aligning the model's output with given consistency criteria.
An automated correctional workflow was introduced: If initial model outputs deviated from required consistency, the system would evaluate both positive and negative prediction consistency, opting for the one with minimized deviation.
The refinement weights, closely mirroring the competition's criteria, ensured the model's results were both accurate and consistent.
