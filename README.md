# Pulmonary_Embolism_Detection
## Problem Description:
Pulmonary embolism (PE) is a serious medical condition caused by an artery blockage in the lung. Its symptoms can often be painful and, in severe cases, life-threatening. Currently, diagnosing PE requires detailed examination of chest CT pulmonary angiography (CTPA) scans, which can comprise hundreds of images per patient. Due to the intricate nature of these scans, accurate identification of PE can be time-consuming and may sometimes lead to overdiagnosis.

With the growing reliance on medical imaging, the workload for radiologists continues to increase. This might lead to potential delays in PE diagnosis and subsequent treatment, raising concerns about patient care. Annually, PE accounts for 60,000-100,000 deaths in the U.S., making timely and precise diagnosis crucial to improve patient outcomes.


### Goal:
We will utilize chest CTPA images, organized as studies, to develop algorithms that can detect and classify PE with enhanced accuracy.
The work can play a pivotal role in minimizing human-induced delays and errors during PE diagnosis and treatment, leading to better patient care and potentially improved outcomes.

### Impact:
As one of the leading causes of cardiovascular deaths, swift and precise PE detection can greatly enhance patient care and potentially save lives. Utilizing machine learning techniques can potentially revolutionize the way we approach PE diagnosis, ensuring patients receive the care they need when they need it.


<table>
  <tr>
    <td><img src="https://github.com/satyajeetburla/Pulmonary_Embolism_Detection/raw/main/image/pe1.jpg" alt="1" width="360px" height="360px"></td>
    <td><img src="https://github.com/satyajeetburla/Pulmonary_Embolism_Detection/raw/main/image/pe2.jpg" alt="2" width="360px" height="360px"></td>
  </tr>
  <tr>
    <td><img src="https://github.com/satyajeetburla/Pulmonary_Embolism_Detection/raw/main/image/pe3.jpg" alt="3" width="360px" height="360px"></td>
    <td><img src="https://github.com/satyajeetburla/Pulmonary_Embolism_Detection/raw/main/image/pe4.png" alt="4" width="360px" height="360px"></td>
  </tr>
</table>

### Find data from the below link : 
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


## Metric: Weighted Log Loss

Exam-level weighted log loss:

Each exam label (9 in total) has its own weight (w_j), and the objective is to predict the probability (p_ij) of each label being present for a given exam.
There's a table that provides the weight for each label, such as Negative for PE, Indeterminate, etc.
Finally we calculates the binary log loss for each label and then computes the mean log loss over all labels.

Image-level weighted log loss:

This is for individual images within each exam. An image either has PE present or doesn't.
The weight for the image-level label is given as w = 0.07361963.
There's a formula provided to compute the log loss for the image-level predictions.

Total Loss:

It's the average of both the image-level and exam-level log losses.
To compute the total loss, you sum the weights of all image rows and exam-level label rows. Then, divide this sum by the total number of rows (both image and exam) to get the average weight. This average weight is then used to adjust the computed log loss.

## Result on Validation
Final Loss : 0.089;  AUC : 0.998



