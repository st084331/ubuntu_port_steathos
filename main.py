""" Import packages """
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
from py_client import client_api
from sklearn.model_selection import StratifiedKFold
from glob import glob

import pydicom as dcm

from monai.transforms import (

    Compose,
    AddChanneld,
    Resized,
    ToTensord,
    Spacingd,
    Orientationd,
    ScaleIntensityRanged,
    CropForegroundd,
)
from monai.data import Dataset, DataLoader
from monai.utils import set_determinism

import pandas as pd


input_dir = './unifesp-fatty-liver'
imaging_dir = os.path.join(input_dir, 'fatty-liver-dataset', 'd_2')

"""
First, we split all patients between Chest and Abdomen. 
We get this information by iterating through the exam directories, opening a dicom file 
and reading the Study Description Tag. 
"""

list_chest_patient_number_dirs = []
list_abdomen_patient_number_dirs = []
for file in (sorted(glob(imaging_dir + '/*' + '/*' + '/*'))):
    dicom = dcm.dcmread(os.path.join(file, os.listdir(file)[0]), stop_before_pixels=True)

    # Here we iterate through each series directory (the last subdirectory) and read the first dicom file
    # (doesn't matter which one you read). We stop before the pixel array, so the process takes less time.
    if dicom[0x0008, 0x1030].value == 'Chest':
        list_chest_patient_number_dirs.append(file.split('/')[4])
    elif dicom[0x0008, 0x1030].value == 'Abdomen':
        list_abdomen_patient_number_dirs.append(file.split('/')[4])

"""
Get the unique values from Chest and Abdomen lists and convert to integers
"""
list_chest_patient_number_dirs = np.unique(list_chest_patient_number_dirs).astype(int)
list_abdomen_patient_number_dirs = np.unique(list_abdomen_patient_number_dirs).astype(int)

"""
Find the index of the patients who have both Chest and Abdomen CTs, and remove from Chest List.
This is important because we are gonna use the Chest List to create the validation set, which should
only contain Chest Scans, because our final goal is to create a model to perform well on these studies.
"""

_, idx, __ = np.intersect1d(list_chest_patient_number_dirs, list_abdomen_patient_number_dirs, assume_unique=True, return_indices=True)
list_only_chest_patient_number_dirs = np.delete(list_chest_patient_number_dirs, idx)


# First we list the patients from the testing set by reading the submission file.
test_split = pd.read_csv(os.path.join(input_dir,'sample_submission.csv'))['Id'].to_numpy()
# Then, we certify that all these patients are chest patients.
common, idx, _ = np.intersect1d(list_only_chest_patient_number_dirs, test_split, assume_unique=True, return_indices=True)

if(len(common) >= 1 and len(test_split) >= 1):
    if all(common == test_split):
        print('All patients from the test set are chest patients.')
    else:
        print('There are patients from the test set which are not chest patients.')
else:
    if common == test_split:
        print('All patients from the test set are chest patients.')
    else:
        print('There are patients from the test set which are not chest patients.')


# Finally, we remove the testing patients.
list_only_chest_patient_number_dirs_wo_test_patients = np.delete(list_only_chest_patient_number_dirs, idx)
print(list_only_chest_patient_number_dirs_wo_test_patients)

n_splits = 3  # change here the number of cross validation groups
n_abd = len(list_abdomen_patient_number_dirs)
n_chest = len(list_only_chest_patient_number_dirs_wo_test_patients)
n_total = n_abd + n_chest
n_val = int(n_chest/n_splits)
percent_test = 100*n_val/n_total

print('CT Abdomen:', n_abd)
print('CT Chest:', n_chest)
print('Total:', n_total)
print('Validation Split:', n_val, 'Percentage of total:', f'{percent_test:.1f}')

labels = pd.read_csv(os.path.join(input_dir, 'train.csv'))  # load the traning csv
labels.index = labels['Id']  # change the index so we can use the loc function to retrieve the Ids
chest_labels = np.array(labels.loc[list_only_chest_patient_number_dirs_wo_test_patients,'ground_truth']) # create an array with only our chest group
splits_dir = './splits'
kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=2022)
i = 0
try:
    os.mkdir(splits_dir)
except FileExistsError:
    print('"splits" directory already exists.')
for train_idx, val_idx in kf.split(list_only_chest_patient_number_dirs_wo_test_patients, chest_labels):
    np.savetxt(os.path.join(splits_dir, f'train_split_{i}.txt'),
               np.concatenate((list_only_chest_patient_number_dirs_wo_test_patients[train_idx], list_abdomen_patient_number_dirs)), delimiter=',', fmt='%d')
    np.savetxt(os.path.join(splits_dir, f'val_split_{i}.txt'), list_only_chest_patient_number_dirs_wo_test_patients[val_idx], delimiter=',',
               fmt='%d')
    i += 1

np.savetxt(os.path.join(splits_dir, f'test_split.txt'), test_split, delimiter=',', fmt='%d')


def load_train_val(split, split_dir):
    """
    This function loads a train and validation split as NumPy arrays, from the txt files we created before.

    'split': this parameter is the split number from the k-fold cross validation. It should be an integer
    between 0 (inclusive) and the k value (not inclusive) you chose.
    'split_dir': the directory where you saved your splits.
    """
    n_splits = 3  # change this to your k-fold number
    if split in range(n_splits):
        train = np.loadtxt(os.path.join(split_dir, f'train_split_{split}.txt'), dtype=int, delimiter=',')
        val = np.loadtxt(os.path.join(split_dir, f'val_split_{split}.txt'), dtype=int, delimiter=',')
        return train, val
    else:
        raise ValueError('Please specify a valid split number.')


def load_test(split_dir):
    """
    This function loads the testing set into a NumPy array.
    'split_dir': the directory where you saved your splits.
    """
    return np.loadtxt(os.path.join(split_dir, 'test_split.txt'), dtype=int, delimiter=',')

class DicomDataset(Dataset):
    def __init__(self, data, transform):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = {'img': None, 'label': None}
        item['img'] = dcm.dcmread(self.data[index]['img'])
        slope = float(item['img'][0x0028, 0x1053].value)
        intercept = float(item['img'][0x0028, 0x1052].value)

        item['img'] = np.array(item['img'].pixel_array, dtype=np.float32)
        item['img'] *= slope
        item['img'] += intercept
        item['label'] = float(self.data[index]['label'])
        if self.transform:
            item = self.transform(item)

        return item


def prepare_train(train, val, imaging_dir, labels_dir, batch_size, pixdim=(1.5, 1.5), a_min=-200,
                  a_max=200, spatial_size=[128, 128]):
    """
    This function creates a training and a validation loader, with all the transforms you want for
    preprocessing. Here, we include some basic transforms, but you can add more operations that you
    find in the Monai documentation.
    https://monai.io/docs.html
    """

    set_determinism(seed=0)
    path_images = sorted(glob(imaging_dir + '/*' + '/*' + '/*' + '/*.dcm'))

    path_train_images = [image for image in path_images if int(image.split('/')[-4]) in train]
    path_val_images = [image for image in path_images if int(image.split('/')[-4]) in val]

    df = pd.read_csv(os.path.join(labels_dir, 'train.csv'), index_col=0)

    train_files = [{"img": image_name, "label": df.loc[label_name, 'ground_truth']} for image_name,
                                                                                        label_name in
                   zip(path_train_images, train)]
    val_files = [{"img": image_name, "label": df.loc[label_name, 'ground_truth']} for image_name,
                                                                                      label_name in
                 zip(path_val_images, val)]

    train_transforms = Compose(
        [
            AddChanneld(keys=["img"]),
            Spacingd(keys=["img"], pixdim=pixdim, mode=("bilinear")),
            ScaleIntensityRanged(keys=["img"], a_min=a_min, a_max=a_max, b_min=0.0, b_max=1.0, clip=True),
            Resized(keys=["img"], spatial_size=spatial_size),
            Orientationd(keys=['img'], axcodes='RA'),
            ToTensord(keys=["img", 'label']),

        ]
    )

    val_transforms = Compose(
        [
            AddChanneld(keys=["img"]),
            Spacingd(keys=["img"], pixdim=pixdim, mode=("bilinear")),
            ScaleIntensityRanged(keys=["img"], a_min=a_min, a_max=a_max, b_min=0.0, b_max=1.0, clip=True),
            Resized(keys=["img"], spatial_size=spatial_size),
            Orientationd(keys=['img'], axcodes='RA'),
            ToTensord(keys=["img", 'label']),

        ]
    )

    train_ds = DicomDataset(data=train_files, transform=train_transforms)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    val_ds = DicomDataset(data=val_files, transform=val_transforms)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=True)

    return train_loader, val_loader


def show_batch(dl):
    """
    This function is to show one batch from your datasets, so that you can si if the it is okay or you need
    to change/delete something.

    `dl`: this parameter should take the data loader, which means you need to prepare first and apply
    the transforms that you want. After that, pass it to this function so that you visualize the patients
    with the transforms that you want.
    """

    for batch_data in dl:
        batch_size = len(batch_data['label'])
        fig, ax = plt.subplots(batch_size // 4, 4, figsize=(16, batch_size))

        for i in range(len(batch_data['label'])):
            label = batch_data['label'][i].item()

            ax[i // 4, i % 4].imshow(batch_data['img'][i, 0], cmap='gray', interpolation='bilinear', aspect='auto')
            ax[i // 4, i % 4].set_title(f'label: {label}')
            ax[i // 4, i % 4].xaxis.set_ticklabels([])
            ax[i // 4, i % 4].yaxis.set_ticklabels([])
        plt.show()
        break

train, val = load_train_val(0, splits_dir)
train_loader, val_loader = prepare_train(train, val, imaging_dir, input_dir, 16, a_min=-200, a_max=200)
#show_batch(train_loader)


client = client_api.AIAAClient(server_url='http://0.0.0.0:5000')
models = client.model_list(label='spleen')
print(models)