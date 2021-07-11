### Reading libraries
import config_file as cfg
# RUNNING THE MODEL
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
import os
import pydicom
import data_input as data_input
from torchvision.transforms.functional import to_tensor
import shutil 
import auxiliar as auxiliar
import subprocess

# check if there are new pictures in the folder
# Copy files from Bucket
os.system('sudo gsutil cp -r  gs://medicine-data-bucket/production/input/* /home/medicine_project/input_data_production/')

# Check files dowloaded
files_to_evaluate = os.listdir("/home/medicine_project/input_data_production/")
#files_to_evaluate = [k.replace("(", "") for k in files_to_evaluate]
#files_to_evaluate = [k.replace(")", "") for k in files_to_evaluate]
#files_to_evaluate = [k.replace(" ", "") for k in files_to_evaluate]     

if len(files_to_evaluate) > 0:
    pneumonia_evaluation = data_input.data_input_prod(files_to_evaluate)
    pneumonia_testloader = torch.utils.data.DataLoader(pneumonia_evaluation, batch_size=1, num_workers=0, collate_fn=auxiliar.collate_fn_prod)

    #if not torch.cuda.is_available():
    #    raise RuntimeError("You should enable GPU runtime.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained = True, min_size = 1024)
    num_classes = 2
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
#model = torch.load("/home/medicine_project/output_data/MODEL")

    model.load_state_dict(torch.load("/home/medicine_project/output_data/MODEL"))
    model = model.eval()

    for i,sample in enumerate(pneumonia_testloader):
    #print(i)
        images =sample
        with torch.no_grad():
            loss_dict = model(images, targets = None)
        detections = loss_dict[0]
        keep_idx = torchvision.ops.nms(detections["boxes"], detections["scores"], cfg.iou_threshold)
        boxes = [b for i, b in enumerate(detections["boxes"]) if i in keep_idx]
        scores = [s for i, s in enumerate(detections["scores"]) if i in keep_idx]
        labels = [l for i, l in enumerate(detections["labels"]) if i in keep_idx]
        #scores_max=max(scores)
        im = to_pil_image(images[0].cpu())
        draw = ImageDraw.Draw(im)
        for box, score, label in zip(boxes, scores, labels):
            if (score >= cfg.score_threshold): #& (score >= score_threshold):
            #print(score)
                coords = box.cpu().tolist()
                draw.rectangle(coords,width=5, outline = "red")
                text = f"{'score:'} {score*100:.2f}%"
                draw.text([coords[0], coords[1]-15], text, fill = "orange")
        name_image = str(files_to_evaluate[i].split(".")[0])
    
        new_folder = "/home/medicine_project/output_production/" + name_image + "/"
        if not os.path.isdir(new_folder):
            os.makedirs(new_folder) 
        im = im.save(new_folder + "diagnosis.png")
        # After this, we can move the photo to a new side
        dest = shutil.move("/home/medicine_project/input_data_production/" + files_to_evaluate[i],
        new_folder + "initial_image.dcm")
        os.system('sudo gsutil cp -r ' + new_folder + '* gs://medicine-data-bucket/production/output/' + name_image)
        # Move from input to processed
        os.system("sudo gsutil mv gs://medicine-data-bucket/production/input/" + name_image + ".dcm gs://medicine-data-bucket/production/processed/" + name_image + ".dcm")







