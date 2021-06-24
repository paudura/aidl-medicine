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



import data_input as data_input

pneumonia_train_dataset = data_input.pneumoniaDataset(train = True, number_validation = 50)
pneumonia_test_dataset = data_input.pneumoniaDataset(train = False, number_validation = 50)

#pneumonia_train_dataset[4][0].shape
def show_image(pil_im):
    plt.imshow(np.asarray(pil_im))

def collate_fn(batch):
    images = []
    targets = []
    for i, t in batch:
        images.append(i)
        targets.append(t)
    return images, targets

pneumonia_trainloader = torch.utils.data.DataLoader(pneumonia_train_dataset, batch_size=16, num_workers=0,collate_fn=collate_fn)
pneumonia_testloader = torch.utils.data.DataLoader(pneumonia_test_dataset, batch_size=1, num_workers=0, collate_fn=collate_fn)


#FasterRCNNModel

if not torch.cuda.is_available():
    raise RuntimeError("You should enable GPU runtime.")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained = True, min_size = 1024)
num_classes = 2
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
model = model.train().to(device)
optimizer = optim.Adam(model.parameters(), weight_decay=0.0001)

# Versio pau V2: FUNCIONA!

model.train()
for batch_idx, sample in enumerate(pneumonia_trainloader):
    #print(batch_idx)
    images, targets = sample[0], sample[1]
    optimizer.zero_grad()
    loss_dict = model(images, targets)
    loss = sum(loss for loss in loss_dict.values())
    loss.backward()
    optimizer.step()
    if batch_idx%2 == 0:
        loss_dict_printable = {k: f"{v.item():.2f}" for k, v in loss_dict.items()}
        print(f"[{batch_idx}/{len(pneumonia_trainloader)}] loss: {loss_dict_printable}")
# Save model
torch.save(model.state_dict(), "/home/medicine_project/output_data/MODEL")


###############
#### TEST #####
###############

def bb_intersection_over_union(boxA, boxB):
    	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0][0], boxB[0])
	yA = max(boxA[0][1], boxB[1])
	xB = min(boxA[0][2], boxB[2])
	yB = min(boxA[0][3], boxB[3])
	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[0][2] - boxA[0][0] + 1) * (boxA[0][3] - boxA[0][1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)
	# return the intersection over union value
	return iou


# Load model
model.load_state_dict(torch.load("/home/medicine_project/output_data/MODEL"))
model = model.eval()
ious = []
for i,sample in enumerate(pneumonia_testloader):
    print(i)
    images,targets=sample[0],sample[1]  
    with torch.no_grad():
        loss_dict = model(images, targets = None)
    detections = loss_dict[0]
    iou_threshold = 0.2
    score_threshold = 0.7
    keep_idx = torchvision.ops.nms(detections["boxes"], detections["scores"], iou_threshold)
    boxes = [b for i, b in enumerate(detections["boxes"]) if i in keep_idx]
    scores = [s for i, s in enumerate(detections["scores"]) if i in keep_idx]
    labels = [l for i, l in enumerate(detections["labels"]) if i in keep_idx]
    scores_max=max(scores)
    im = to_pil_image(images[0].cpu())
    draw = ImageDraw.Draw(im)
    #for box, score, label in zip(boxes, scores, labels):
    #    if (score >= score_threshold): #& (score >= score_threshold):
            #print(score)
    #        coords = box.cpu().tolist()
    #        draw.rectangle(coords,width = 5)
    # Metric
    boxes_max = boxes[scores == scores_max]
    iou = bb_intersection_over_union((targets[0]["boxes"]).tolist(), boxes_max.tolist())
    # coords
    coords = boxes_max.cpu().tolist()
    draw.rectangle(coords,width=5, outline = "red")
    coords2 = targets[0]["boxes"][0].cpu().tolist()
    draw.rectangle(coords2,width=5, outline = "green")
    text = f"{'iou:'} {iou*100:.2f}%"
    draw.text([coords2[0], coords2[1]-15], text, fill = "orange")
    show_image(im)
### GUARDAR LA IMAGEN EN LA CARPETA OUTPUT
    im = im.save("/home/medicine_project/output_data/prediction_" + str(i) + ".png")
    ious.append(iou)

print(sum(ious) / len(ious))

# NEW IMAGE EVALUATION


# Data transformation de la nova imatge
# printarla





