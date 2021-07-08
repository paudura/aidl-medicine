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
        # juntamos por paciente
        image_bbox_df.reset_index(drop = True, inplace = True)
        parametros = {}

        for i in range(image_bbox_df.shape[0]):
            if image_bbox_df.loc[i, "patientId"] not in parametros.keys():
                parametros[image_bbox_df.loc[i, "patientId"]] = {}

                patient_id = image_bbox_df.loc[i, "patientId"]
                data_patients = image_bbox_df.loc[image_bbox_df["patientId"] == patient_id, :]
                parametros[image_bbox_df.loc[i, "patientId"]]["x"] = [r for r in data_patients["x"]]
                parametros[image_bbox_df.loc[i, "patientId"]]["y"] = [r for r in data_patients["y"]]
                parametros[image_bbox_df.loc[i, "patientId"]]["height"] = [r for r in data_patients["height"]]
                parametros[image_bbox_df.loc[i, "patientId"]]["width"] = [r for r in data_patients["width"]]
                if i % 1000 == 0: 
                    print(str(i) + " rows (", str(round(i/image_bbox_df.shape[0]*100, 2)) + "%)")
        image_bbox_df.drop_duplicates(subset = "patientId", keep = "first", inplace = True)
        image_bbox_df.reset_index(drop = True, inplace = True)          

        return image_bbox_df, parametros

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
        dd, parametros = train_dataframe()
        dd = dd.loc[~dd['path'].isna(),:]
        dd.reset_index(drop = True, inplace = True)
        dd = dd.loc[dd["Target"] == 1, :]
        dd.reset_index(drop = True, inplace = True)
        #random.shuffle(dd        

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
            self.annotations.append(parametros[dd.loc[i, "patientId"]])


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

        for q in range(len(self.annotations[idx]["x"])):
            
            #for _, annotation in self.annotations[idx]:
            x = self.annotations[idx]["x"][q]
            y = self.annotations[idx]["y"][q]
            width = self.annotations[idx]["width"][q]
            heigth = self.annotations[idx]["height"][q]
            
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



class data_input_classification():
    def __init__(self, files_to_evaluate):
        self.files_to_evaluate = files_to_evaluate
        self.imgs = []
        for k in self.files_to_evaluate:
            self.imgs.append(k)

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




class pneumoniaDataset_new(object):
    def __init__(self, train = True, validation = False, number_validation = 20, number_test = 50, device = "cpu"):
        # load all image files, sorting them to
        # ensure that they are aligned
        self.train = train
        self.validation = validation
        self.number_validation = number_validation
        self.number_test = number_test
        dd, parametros = train_dataframe()
        dd = dd.loc[~dd['path'].isna(),:]
        dd.reset_index(drop = True, inplace = True)
        dd = dd.loc[dd["Target"] == 1, :]
        dd.reset_index(drop = True, inplace = True)
        #random.shuffle(dd        

        if self.train:
            dd = dd.loc[(number_validation + number_test):]
            dd.reset_index(drop = True, inplace = True)

        elif self.validation:
            dd = dd.loc[: number_validation]
            dd.reset_index(drop = True, inplace = True)
        else:
            # create validation dataset
            dd2 = dd.loc[number_validation : (number_validation + number_test)]
            dd.reset_index(drop = True, inplace = True)

        self.imgs = []
        self.annotations = []

        for i in range(dd.shape[0]):
            filename = dd.loc[i, "path"] 
            self.imgs.append(filename)
            self.annotations.append(parametros[dd.loc[i, "patientId"]])


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

        for q in range(len(self.annotations[idx]["x"])):
            
            #for _, annotation in self.annotations[idx]:
            x = self.annotations[idx]["x"][q]
            y = self.annotations[idx]["y"][q]
            width = self.annotations[idx]["width"][q]
            heigth = self.annotations[idx]["height"][q]
            
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