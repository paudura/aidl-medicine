
##%%


import data_input as di
import auxiliar as aux
import torchvision as tv
import pneumonia_dataset as pneu_data
from torch.utils.data import DataLoader
import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
import auxiliar as aux

train_dataframe = di.train_dataframe()
test_dataframe = di.test_dataframe()
# Check
train_dataframe = train_dataframe.loc[ ~ train_dataframe["path"].isna(), :]
train_dataframe.reset_index(drop = True, inplace = True)
train_dataframe.reset_index(drop = False, inplace = True)

# Split train - validation
validation_frac = 0.2

# select validation and train
train_ids = train_dataframe["index"].sample(frac = 1 - validation_frac, random_state = 42)
validation_ids = train_dataframe.loc[~train_dataframe["index"].isin(train_ids), "index"]



root = "/home/medicine_project/input_data/"
subset = "train"
pIds_train = list(train_dataframe.loc[train_dataframe["index"].isin(train_ids), "patientId"])
pIds_val = list(train_dataframe.loc[train_dataframe["index"].isin(validation_ids), "patientId"])
test_ids = list(test_dataframe["patientId"])
rescale_factor  = 4

# where are the boxes
pId_boxes_dict = {}
for pId in train_dataframe.loc[(train_dataframe['Target']==1)]['patientId'].unique().tolist():
    pId_boxes_dict[pId] = aux.get_boxes_per_patient(train_dataframe, pId)

# TBD add normalization of images into transforms
# define transformation 
transform = tv.transforms.Compose([tv.transforms.ToTensor()])
rotation_angle = 3
warping = True


dataset_train = pneu_data.PneumoniaDataset(root = root, subset = 'train', pIds= pIds_train, predict=False, 
                                 boxes=pId_boxes_dict, rescale_factor=rescale_factor, transform=transform,
                                 rotation_angle= rotation_angle, warping = warping)

                                 
dataset_validation = pneu_data.PneumoniaDataset(root = root, subset = 'train', pIds= pIds_val, predict=False, 
                                 boxes=pId_boxes_dict, rescale_factor=rescale_factor, transform=transform,
                                 rotation_angle = 0, warping = False)

                                 
dataset_test = pneu_data.PneumoniaDataset(root = root, subset = 'test', pIds= test_ids, predict=True, 
                                 boxes=None, rescale_factor=rescale_factor, transform=transform,
                                 rotation_angle = 0, warping = False)

batch_size = 3
loader_train = DataLoader(dataset=dataset_train,
                           batch_size=batch_size,
                           shuffle=True) 


loader_valid = DataLoader(dataset = dataset_validation,
                           batch_size=batch_size,
                           shuffle=True) 

loader_test = DataLoader(dataset=dataset_test,
                         batch_size=batch_size,
                         shuffle=False)


# Check if train images have been properly loaded
print('{} images in train set, {} images in validation set, and {} images in test set.'.format(len(dataset_train),
                                                                                               len(dataset_validation),
                                                                                               len(dataset_test)))
img_batch, target_batch, pId_batch = next(iter(loader_train))
print('Tensor batch size:', img_batch.size())


for i in np.random.choice(len(dataset_train), size=5, replace=False):
    img, target, pId = dataset_train[i] # picking an image with pneumonia
    print('\nImage and mask shapes:', img.shape, target.shape)
    print('Patient ID:', pId)
    print('Image scale: {} - {}'.format(img[0].min(), img[0].max()))
    print('Target mask scale: {} - {}'.format(target[0].min(), target[0].max()))
    plt.imshow(img[0], cmap=mpl.cm.gist_gray) # [0] is the channel index (here there's just one channel)
    plt.imshow(target[0], cmap=mpl.cm.jet, alpha=0.2)
    plt.axis('off')
    plt.show()



# Check if test images have been properly loaded

img, pId = dataset_test[0] 
print('Image shape:', img.shape)
print('Patient ID:', pId)
print('Image scale: {} - {}'.format(img[0].min(), img[0].max()))
plt.imshow(img[0], cmap=mpl.cm.gist_gray) # [0] is the channel index (here there's just one channel)





