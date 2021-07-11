## SCRIPT TO FIND OUT THE CLASSIFICATION METRIX
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
import data_input as data_input
import config_file as cfg
import auxiliar as auxiliar
import torchvision
from numpy.core.fromnumeric import size
import torch
import torchvision
import torch.optim as optim
from torchvision.transforms.functional import to_pil_image
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import numpy as np
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import  FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
import time
import auxiliar as auxiliar
import data_input as data_input

dd, parametros = data_input.train_dataframe()
dd = dd.loc[~dd['path'].isna(),:]
dd.reset_index(drop = True, inplace = True)
dd = dd.loc[~dd['Target'].isna(),:]
dd.reset_index(drop = True, inplace = True)
dd = dd.loc[:200,]
dd.reset_index(drop = True, inplace = True)

TP = 0
TN = 0
FP = 0
FN = 0

files_to_evaluate = list(dd["path"])
pneumonia_evaluation = data_input.data_input_classification(files_to_evaluate)
pneumonia_testloader = torch.utils.data.DataLoader(pneumonia_evaluation, batch_size=1, num_workers=0, collate_fn=auxiliar.collate_fn_prod)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained = True, min_size = 1024)
num_classes = 2
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
model.load_state_dict(torch.load("/home/medicine_project/output_data/MODEL"))
model = model.eval()

d1 = {'thres':[], 'precision':[] , 'recall':[], 'accuracy':[]}
matrix = pd.DataFrame(d1)

scores_thres = [0.1, 0.2, 0.3, 0.4, 0.5]
fila = 0
for t in scores_thres:
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for q,sample in enumerate(pneumonia_testloader):
        #print(q)
        images = sample
        with torch.no_grad():
            loss_dict = model(images, targets = None)
        detections = loss_dict[0]
        keep_idx = torchvision.ops.nms(detections["boxes"], detections["scores"], cfg.iou_threshold)
        boxes = [b for q, b in enumerate(detections["boxes"]) if q in keep_idx]
        scores = [s for q, s in enumerate(detections["scores"]) if q in keep_idx]
        labels = [l for q, l in enumerate(detections["labels"]) if q in keep_idx]
    #scores_max=max(scores)
        try:
            if max(scores) > t: 
                test = 1
            else:
                test = 0
        except:
            test = 0
        #  values
        if (test == 1) & (dd.loc[q, "Target"] == 0):
            FP = FP + 1
        elif (test == 1) & (dd.loc[q, "Target"] == 1):
            TP = TP + 1
        elif (test == 0) & (dd.loc[q, "Target"] == 0):
                TN = TN + 1
        elif (test == 0) & (dd.loc[q, "Target"] == 1):
                FN = FN + 1
        
        if q % 20 == 0: 
            print(str(q) + " rows (", str(round(q/dd.shape[0]*100, 2)) + "%)")


    RECALL = TP / (TP + FN)
    PRECISION = TP / (TP + FP)
    ACCURACY = (TP + TN) / (TP + FP + TN + FN)

    matrix.loc[fila, "thres"] = t
    matrix.loc[fila, "precision"] = PRECISION
    matrix.loc[fila, "recall"] = RECALL
    matrix.loc[fila, "accuracy"] = ACCURACY
    fila = fila + 1
    

    

    

    
