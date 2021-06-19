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

################
## INPUT DATA ##
################

def pneumonia_locations():
    # empty dictionary
    pneumonia_locations = {}
    # load table
    with open(os.path.join('/home/medicine_project/input_data/stage_2_train_labels.csv'), mode='r') as infile:
        # open reader
        reader = csv.reader(infile)
        # skip header
        next(reader, None)
        # loop through rows
        for rows in reader:
            # retrieve information
            filename = rows[0]
            location = rows[1:5]
            pneumonia = rows[5]
            # if row contains pneumonia add label to dictionary
            # which contains a list of pneumonia locations per filename
        if pneumonia == '1':
            # convert string to float to int
            location = [int(float(i)) for i in location]
            # save pneumonia location in dictionary
            if filename in pneumonia_locations:
                pneumonia_locations[filename].append(location)
            else:
                pneumonia_locations[filename] = [location]
    return(pneumonia_locations)


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
        image_bbox_df = pd.merge(comb_box_df, image_df, on='patientId', how='left').sort_values('patientId')
        #print(image_bbox_df.shape[0], 'image bounding boxes')
        #print(sum(image_bbox_df["path"].isna()))
        return(image_bbox_df)

def test_dataframe():
        dicom_dir = '/home/medicine_project/input_data/stage_2_test_images/'
        image_df = pd.DataFrame({'path': glob(os.path.join(dicom_dir, '*.dcm'))})
        image_df['patientId'] = image_df['path'].map(lambda x: os.path.splitext(os.path.basename(x))[0])
        print(image_df.shape[0], 'images found')
        return(image_df)


#dd = train_dataframe()
#print(dd.head(n =10))
#print(dd.columns)
#print(dd.shape)



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
        ds = pydicom.read_file(img_path) #no llegeix 
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
            px = [x, x - width/2, x + width/2]
            py = [y, y - heigth/2, y + heigth/2]

        poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)] #not necessary
        poly = [p for x in poly for p in x]
            
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



#pneumonia_train_dataset = pneumoniaDataset(train = True)
#pneumonia_test_dataset = pneumoniaDataset(train = False)

#def collate_fn(batch):
#    images = []
#    targets = []
#    for i, t in batch:
#        images.append(i)
#        targets.append(t)
#    return images, targets

#pneumonia_trainloader = torch.utils.data.DataLoader(pneumonia_train_dataset, batch_size=4, num_workers=0,collate_fn=collate_fn)
#pneumonia_testloader = torch.utils.data.DataLoader(pneumonia_test_dataset, batch_size=1, num_workers=0, collate_fn=collate_fn)


