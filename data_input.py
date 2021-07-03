import pandas as pd
import numpy
import csv
import os 
from glob import glob
import json
from torchvision.transforms.functional import to_tensor
from PIL import Image, ImageDraw
import torch
import pydicom
import numpy as np
import copy
import random
torch.device("cuda" if torch.cuda.is_available() else "cpu")
import matplotlib.pyplot as plt
import pydicom as dcm
from matplotlib.patches import Rectangle

################
## INPUT DATA ##
################

def train_dataframe():
        det_class_path = '/home/medicine_project/input_data/stage_2_detailed_class_info.csv'
        bbox_path = '/home/medicine_project/input_data/stage_2_train_labels.csv'
        dicom_dir = '/home/medicine_project/input_data/stage_2_train_images/'
        det_class_df = pd.read_csv(det_class_path)
        bbox_df = pd.read_csv(bbox_path)
        comb_box_df = pd.concat([bbox_df, det_class_df.drop('patientId',1)], 1)

        image_df = pd.DataFrame({'path': glob(os.path.join(dicom_dir, '*.dcm'))})
        image_df['patientId'] = image_df['path'].map(lambda x: os.path.splitext(os.path.basename(x))[0])
        #print(image_df.shape[0], 'images found')
        img_pat_ids = set(image_df['patientId'].values.tolist())
        box_pat_ids = set(comb_box_df['patientId'].values.tolist())
        # check to make sure there is no funny business
        #assert img_pat_ids.union(box_pat_ids)==img_pat_ids, "Patient IDs should be the same"
        image_bbox_df = pd.merge(comb_box_df, image_df, on='patientId', how='left').sort_values('patientId').reset_index(drop = True)
        #print(image_bbox_df.shape[0], 'image bounding boxes')
        #print(sum(image_bbox_df["path"].isna()))
        return(image_bbox_df)

def test_dataframe():
        dicom_dir = '/home/medicine_project/input_data/stage_2_test_images/'
        image_df = pd.DataFrame({'path': glob(os.path.join(dicom_dir, '*.dcm'))})
        image_df['patientId'] = image_df['path'].map(lambda x: os.path.splitext(os.path.basename(x))[0])
        print(image_df.shape[0], 'images found')
        return(image_df)


class pneumoniaDataset(object):
    def __init__(self, train = True, number_validation = 20, device = "cpu"):
        # load all image files, sorting them to
        # ensure that they are aligned
        self.train = train
        self.number_validation = number_validation
        dd = train_dataframe()
        dd = dd.loc[~dd['path'].isna(),:]
        dd.reset_index(drop = True, inplace = True)
        dd = dd.loc[dd["Target"] == 1, :]
        dd.reset_index(drop = True, inplace = True)
        #random.shuffle(dd)             

        if self.train:
            dd = dd.loc[number_validation:]
            dd.reset_index(drop = True, inplace = True)
        else:
            # create validation dataset
            dd = dd.loc[:number_validation]
            dd.reset_index(drop = True, inplace = True)

        self.imgs = []
        self.annotations = []

        for i in range(dd.shape[0]):
            filename = dd.loc[i, "path"] 
            self.imgs.append(filename)
            self.annotations.append(dd.loc[i, ["x", "y", "height", "width"]])


    def __getitem__(self, idx, device = "cpu"):
        # load images
        img_path = self.imgs[idx]
        ds = pydicom.read_file(img_path)
        image = ds.pixel_array
        # If grayscale. Convert to RGB for consistency.
        if len(image.shape) != 3 or image.shape[2] != 3:
            image = np.stack((image,) * 3, -1)

        # get bounding box coordinates for each mask
        boxes = []
        #for _, annotation in self.annotations[idx]:
        x  = self.annotations[idx]["x"]
        y = self.annotations[idx]["y"]
        width = self.annotations[idx]["width"]
        heigth = self.annotations[idx]["height"]
            
        if np.isnan(x):
            px = []
            py = []
        else:
            px = [x, x + width]
            py = [y, y + heigth]
            
        if np.isnan(x):
            boxes.append([])
        else:
            boxes.append([min(px), min(py), max(px), max(py)]) 
        #select the corners of the boxes for each axis. it should be a list with 4 values: 2 coordinates.
        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32, device = device)
        # there is only one class
        labels = torch.ones((len(boxes),), dtype=torch.int64, device = device)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels

        image = to_tensor(image).to(device)
        
        return image, target

    def __len__(self):
        return len(self.imgs)

def show_dicom_images_with_boxes(dd):
    f, ax = plt.subplots(2,2, figsize=(16,18))
    for i  in range(dd.shape[0]):
        patientImage = dd.loc[i, 'patientId']+'.dcm'
        imagePath = os.path.join("/home/medicine_project/input_data/stage_2_train_images/",patientImage)
        data_row_img_data = dcm.read_file(imagePath)
        modality = data_row_img_data.Modality
        age = data_row_img_data.PatientAge
        sex = data_row_img_data.PatientSex
        data_row_img = dcm.dcmread(imagePath)
        ax[i//3, i%3].imshow(data_row_img.pixel_array, cmap=plt.cm.bone) 
        ax[i//3, i%3].axis('off')
        ax[i//3, i%3].set_title('ID: {}\nModality: {} Age: {} Sex: {} Target: {}\nClass: {}'.format(
                dd.loc[i, 'patientId'],modality, age, sex, dd.loc[i, 'Target'], dd.loc[i, 'class']))
        

        ax[i//3, i%3].add_patch(Rectangle(xy=(dd.loc[i, "x"], dd.loc[i, "y"]),
                    width=dd.loc[i, "width"],height=dd.loc[i, "height"], 
                    color="yellow",alpha = 0.1))  

        circle1 = plt.Circle((dd.loc[i, "x"], dd.loc[i, "y"]), 0.5, color='r')

        ax[i//3, i%3].add_patch(circle1)
        plt.show()
        plt.savefig('books_read.png')

class data_input_prod():
    def __init__(self, files_to_evaluate):
        self.files_to_evaluate = files_to_evaluate
        self.imgs = []
        path0 = "/home/medicine_project/input_data_production/"
        for k in self.files_to_evaluate:
            filename = path0 + k
            self.imgs.append(filename)

    def __getitem__(self, idx, device = "cpu"):
        img_path = self.imgs[idx]
        ds = pydicom.read_file(img_path)
        image = ds.pixel_array
        if len(image.shape) != 3 or image.shape[2] != 3:
            image = np.stack((image,) * 3, -1)
        image = to_tensor(image).to(device)
        return image

    def __len__(self):
        return len(self.imgs)


