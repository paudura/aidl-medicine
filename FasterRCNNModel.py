# RUNNING THE MODEL
import torch
import torchvision
import torch.optim as optim

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import  FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator



import data_input as data_input

pneumonia_train_dataset = data_input.pneumoniaDataset(train = True)
pneumonia_test_dataset = data_input.pneumoniaDataset(train = False)

#pneumonia_train_dataset[4][0].shape


def collate_fn(batch):
    images = []
    targets = []
    for i, t in batch:
        images.append(i)
        targets.append(t)
    return images, targets

pneumonia_trainloader = torch.utils.data.DataLoader(pneumonia_train_dataset, batch_size=3, num_workers=0,collate_fn=collate_fn)
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

# Versio Ines
#Model train
#model.train()
#for images, targets in enumerate(pneumonia_trainloader):
#    #print(images)
#    optimizer.zero_grad()
#    predictions, loss_dict = model(images, targets)
#    loss = sum(loss for loss in loss_dict.values())
#    loss.backward()

#    if i%2 == 0:
#        loss_dict_printable = {k: f"{v.item():.2f}" for k, v in loss_dict.items()}
#        print(f"[{i}/{len(pneumonia_trainloader)}] loss: {loss_dict_printable}")



# Versio pau v1
#for i, (images, targets) in enumerate(pneumonia_trainloader):
#    print (i)
#    optimizer.zero_grad()
#    print("optimizer zero grad OK")
#    predictions, loss_dict = model(images, targets)
#    print("model OK")
#    loss = sum(loss for loss in loss_dict.values())
#    print("Loss OK")
#    loss.backward()
#    print("Loss OK2")
#    optimizer.step()
#    print("optimizer OK")

# Versio pau V2: FUNCIONA!

model.train()
for batch_idx, sample in enumerate(pneumonia_trainloader):
    #print(batch_idx)
    images, targets = sample[0], sample[1]
    #print("OK1")
    optimizer.zero_grad()
    #print("OK2")
    loss_dict = model(images, targets)
    #print("OK3")
    loss = sum(loss for loss in loss_dict.values())
    #print("OK4")
    loss.backward()
    #print("OK5")
    optimizer.step()
    if batch_idx%2 == 0:
        loss_dict_printable = {k: f"{v.item():.2f}" for k, v in loss_dict.items()}
        print(f"[{batch_idx}/{len(pneumonia_trainloader)}] loss: {loss_dict_printable}")



