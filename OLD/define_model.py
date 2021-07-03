
from torch import nn
import torch
import torch.nn.functional as F
import numpy as np
import skimage
import shutil
import time
from torch.autograd import Variable

# Define the nn convolutional block
class conv_block(nn.Module):
    """
    Define the [convolution - batch normalization - activation] block 
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True,
                 bn_momentum=0.9, alpha_leaky=0.03):
        super(conv_block, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-05, momentum=bn_momentum)
        self.activ = nn.LeakyReLU(negative_slope=alpha_leaky)

    def forward(self, x):
        return self.activ(self.bn(self.conv(x)))
    
# Define the nn transposed convolutional block
class conv_t_block(nn.Module):
    """
    Define the [convolution_transpose - batch normalization - activation] block 
    """

    def __init__(self, in_channels, out_channels, output_size=None, kernel_size=3, bias=True,
                 bn_momentum=0.9, alpha_leaky=0.03):
        super(conv_t_block, self).__init__()
        self.conv_t = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=2, padding=1, 
                                         bias=bias)
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-05, momentum=bn_momentum)
        self.activ = nn.LeakyReLU(negative_slope=alpha_leaky)

    def forward(self, x, output_size):
        return self.activ(self.bn(self.conv_t(x, output_size=output_size)))    



class PneumoniaUNET(nn.Module):
    
    def __init__(self):
        super(PneumoniaUNET, self).__init__()
        
        self.down_1 = nn.Sequential(conv_block(in_channels=1, out_channels=64), conv_block(in_channels=64, out_channels=64))
        self.down_2 = nn.Sequential(conv_block(in_channels=64, out_channels=128), conv_block(in_channels=128, out_channels=128))
        self.down_3 = nn.Sequential(conv_block(in_channels=128, out_channels=256), conv_block(in_channels=256, out_channels=256))
        self.down_4 = nn.Sequential(conv_block(in_channels=256, out_channels=512), conv_block(in_channels=512, out_channels=512))
        self.down_5 = nn.Sequential(conv_block(in_channels=512, out_channels=512), conv_block(in_channels=512, out_channels=512))

        self.middle = nn.Sequential(conv_block(in_channels=512, out_channels=512), conv_block(in_channels=512, out_channels=512))
        self.middle_t = conv_t_block(in_channels=512, out_channels=256)

        self.up_5 = nn.Sequential(conv_block(in_channels=768, out_channels=512), conv_block(in_channels=512, out_channels=512))
        self.up_5_t = conv_t_block(in_channels=512, out_channels=256)
        self.up_4 = nn.Sequential(conv_block(in_channels=768, out_channels=512), conv_block(in_channels=512, out_channels=512))
        self.up_4_t = conv_t_block(in_channels=512, out_channels=128)
        self.up_3 = nn.Sequential(conv_block(in_channels=384, out_channels=256), conv_block(in_channels=256, out_channels=256))
        self.up_3_t = conv_t_block(in_channels=256, out_channels=64)
        self.up_2 = nn.Sequential(conv_block(in_channels=192, out_channels=128), conv_block(in_channels=128, out_channels=128))
        self.up_2_t = conv_t_block(in_channels=128, out_channels=32)
        self.up_1 = nn.Sequential(conv_block(in_channels=96, out_channels=64), conv_block(in_channels=64, out_channels=1))
        
    def forward(self, x):
        down1 = self.down_1(x) # (1x256x256 -> 64x256x256)
        out = F.max_pool2d(down1, kernel_size=2, stride=2) # (64x256x256 -> 64x128x128)

        down2 = self.down_2(out) # (64x128x128 -> 128x128x128)
        out = F.max_pool2d(down2, kernel_size=2, stride=2) # (128x128x128 -> 128x64x64)

        down3 = self.down_3(out) # (128x64x64 -> 256x64x64)
        out = F.max_pool2d(down3, kernel_size=2, stride=2) # (256x64x64 -> 256x32x32)

        down4 = self.down_4(out) # (256x32x32 -> 512x32x32)
        out = F.max_pool2d(down4, kernel_size=2, stride=2) # (512x32x32 -> 512x16x16)

        down5 = self.down_5(out) # (512x16x16 -> 512x16x16)
        out = F.max_pool2d(down5, kernel_size=2, stride=2) # (512x16x16 -> 512x8x8)

        out = self.middle(out) # (512x8x8 -> 512x8x8)
        out = self.middle_t(out, output_size=down5.size()) # (512x8x8 -> 256x16x16)

        out = torch.cat([down5, out], 1) # (512x16x16-concat-256x16x16 -> 768x16x16)
        out = self.up_5(out) # (768x16x16 -> 512x16x16)
        out = self.up_5_t(out, output_size=down4.size()) # (512x16x16 -> 256x32x32)

        out = torch.cat([down4, out], 1) # (512x32x32-concat-256x32x32 -> 768x32x32)
        out = self.up_4(out) # (768x32x32 -> 512x32x32)
        out = self.up_4_t(out, output_size=down3.size()) # (512x32x32 -> 128x64x64)
        
        out = torch.cat([down3, out], 1) # (256x64x64-concat-128x64x64 -> 384x64x64)
        out = self.up_3(out) # (384x64x64 -> 256x64x64)
        out = self.up_3_t(out, output_size=down2.size()) # (256x64x64 -> 64x128x128)
        
        out = torch.cat([down2, out], 1) # (128x128x128-concat-64x128x128 -> 192x128x128)
        out = self.up_2(out) # (192x128x128 -> 128x128x128)
        out = self.up_2_t(out, output_size=down1.size()) # (128x128x128 -> 32x256x256)
        
        out = torch.cat([down1, out], 1) # (64x256x256-concat-32x256x256 -> 96x256x256)
        out = self.up_1(out) # (96x256x256 -> 1x256x256)
        
        return out    



# Create the loss function
# Define the 2D Sigmoid + Binary Cross Entropy loss function BCEWithLogitsLoss
# TBD add weights for unbalanced class
# NOTE: Rather than using weights, I also intended to try implement focal loss (see RetinaNet)
class BCEWithLogitsLoss2d(nn.Module):

    def __init__(self, weight=None, size_average=True):
        super(BCEWithLogitsLoss2d, self).__init__()
        self.loss = nn.BCEWithLogitsLoss(weight, size_average)

    def forward(self, scores, targets):
        scores_flat = scores.view(-1)
        targets_flat = targets.view(-1)
        return self.loss(scores_flat, targets_flat)


# Define auxiliary metric functions

# define function that creates a square mask for a box from its coordinates 
def box_mask(box, shape=1024):
    """
    :param box: [x, y, w, h] box coordinates
    :param shape: shape of the image (default set to maximum possible value, set to smaller to save memory)
    :returns: (np.array of bool) mask as binary 2D array
    """
    x, y, w, h = box
    mask = np.zeros((shape, shape), dtype=bool)
    mask[y:y+h, x:x+w] = True 
    return mask

# # debug code for above function
# plt.imshow(box_mask([5,20,50,100], shape=256), cmap=mpl.cm.jet)

# define function that extracts confidence and coordinates of boxes from a prediction mask
def parse_boxes(msk, threshold=0.20, connectivity=None):
    """
    :param msk: (torch.Tensor) CxWxH tensor representing the prediction mask
    :param threshold: threshold in the range 0-1 above which a pixel is considered a positive target
    :param connectivity: connectivity parameter for skimage.measure.label segmentation (can be None, 1, or 2)
                         http://scikit-image.org/docs/dev/api/skimage.measure.html#skimage.measure.label
    :returns: (list, list) predicted_boxes, confidences
    """
    # extract 2d array
    msk = msk[0]
    # select pixels above threshold and mark them as positives (1) in an array of equal size as the input prediction mask
    pos = np.zeros(msk.shape)
    pos[msk>threshold] = 1.
    # label regions
    lbl = skimage.measure.label(pos, connectivity=connectivity)
    
    predicted_boxes = []
    confidences = []
    # iterate over regions and extract box coordinates
    for region in skimage.measure.regionprops(lbl):
        # retrieve x, y, height and width and add to prediction string
        y1, x1, y2, x2 = region.bbox
        h = y2 - y1
        w = x2 - x1
        c = np.nanmean(msk[y1:y2, x1:x2])
        # add control over box size (eliminate if too small)
        min_box_area = 10000
        if w*h > min_box_area: 
            predicted_boxes.append([x1, y1, w, h])
            confidences.append(c)
    
    return predicted_boxes, confidences

# # debug code for above function
# plt.imshow(dataset_train[3][1][0], cmap=mpl.cm.jet) 
# print(dataset_train[3][1].shape)
# print(parse_boxes(dataset_train[3][1]))

# define function that creates prediction strings as expected in submission
def prediction_string(predicted_boxes, confidences):
    """
    :param predicted_boxes: [[x1, y1, w1, h1], [x2, y2, w2, h2], ...] list of predicted boxes coordinates 
    :param confidences: [c1, c2, ...] list of confidence values for the predicted boxes
    :returns: prediction string 'c1 x1 y1 w1 h1 c2 x2 y2 w2 h2 ...'
    """
    prediction_string = ''
    for c, box in zip(confidences, predicted_boxes):
        prediction_string += ' ' + str(c) + ' ' + ' '.join([str(b) for b in box])
    return prediction_string[1:]   

# # debug code for above function
# predicted_boxes, confidences = parse_boxes(dataset_train[3][1])
# print(predicted_boxes, confidences)
# print(prediction_string(predicted_boxes, confidences))

# define iou function
def IoU(pr, gt):
    """
    :param pr: (numpy_array(bool)) prediction array 
    :param gt: (numpy_array(bool)) ground truth array 
    :returns: IoU (pr, gt) = intersection (pr, gt) / union (pr, gt)
    """
    IoU = (pr & gt).sum() / ((pr | gt).sum() + 1.e-9)
    return IoU

# # debug code for above function
# pr = box_mask([50,60,100,150], shape=256)
# gt = box_mask([30,40,100,140], shape=256)
# plt.imshow(pr, cmap=mpl.cm.Reds, alpha=0.3)
# plt.imshow(gt, cmap=mpl.cm.Greens, alpha=0.3)
# print(IoU(pr, gt))

# define precision function
def precision(tp, fp, fn):
    """
    :param tp: (int) number of true positives
    :param fp: (int) number of false positives
    :param fn: (int) number of false negatives
    :returns: precision metric for one image at one threshold
    """
    return float(tp) / (tp + fp + fn + 1.e-9)

# # debug code for above function
# print(precision(3,1,1))

# define function that calculates the average precision of an image
def average_precision_image(predicted_boxes, confidences, target_boxes, shape=1024):
    """
    :param predicted_boxes: [[x1, y1, w1, h1], [x2, y2, w2, h2], ...] list of predicted boxes coordinates 
    :param confidences: [c1, c2, ...] list of confidence values for the predicted boxes
    :param target_boxes: [[x1, y1, w1, h1], [x2, y2, w2, h2], ...] list of target boxes coordinates 
    :param shape: shape of the boolean masks (default set to maximum possible value, set to smaller to save memory)
    :returns: average_precision
    """
    
    # if both predicted and target boxes are empty, precision is NaN (and doesn't count towards the batch average)
    if predicted_boxes == [] and target_boxes == []:
        return np.nan
    else:
        # if we have predicted boxes but no target boxes, precision is 0
        if len(predicted_boxes)>0 and target_boxes == []:
            return 0.0
        # if we have target boxes but no predicted boxes, precision is 0
        elif len(target_boxes)>0 and predicted_boxes == []:
            return 0.0
        # if we have both predicted and target boxes, proceed to calculate image average precision
        else:
            # define list of thresholds for IoU [0.4 , 0.45, 0.5 , 0.55, 0.6 , 0.65, 0.7 , 0.75]
            thresholds = np.arange(0.4, 0.8, 0.05) 
            # sort boxes according to their confidence (from largest to smallest)
            predicted_boxes_sorted = list(reversed([b for _, b in sorted(zip(confidences, predicted_boxes), 
                                                                         key=lambda pair: pair[0])]))            
            average_precision = 0.0
            for t in thresholds: # iterate over thresholds
                # with a first loop we measure true and false positives
                tp = 0 # initiate number of true positives
                fp = len(predicted_boxes) # initiate number of false positives 
                for box_p in predicted_boxes_sorted: # iterate over predicted boxes coordinates
                    box_p_msk = box_mask(box_p, shape) # generate boolean mask
                    for box_t in target_boxes: # iterate over ground truth boxes coordinates
                        box_t_msk = box_mask(box_t, shape) # generate boolean mask
                        iou = IoU(box_p_msk, box_t_msk) # calculate IoU
                        if iou>t:
                            tp += 1 # if IoU is above the threshold, we got one more true positive
                            fp -= 1 # and one less false positive
                            break # proceed to the next predicted box
                # with a second loop we measure false negatives
                fn = len(target_boxes) # initiate number of false negatives
                for box_t in target_boxes: # iterate over ground truth boxes coordinates
                    box_t_msk = box_mask(box_t, shape) # generate boolean mask
                    for box_p in predicted_boxes_sorted: # iterate over predicted boxes coordinates
                        box_p_msk = box_mask(box_p, shape) # generate boolean mask
                        iou = IoU(box_p_msk, box_t_msk) # calculate IoU
                        if iou>t:
                            fn -= 1
                            break # proceed to the next ground truth box
                # TBD: this algo must be checked against the official Kaggle evaluation method which is still not clear...
                average_precision += precision(tp, fp, fn) / float(len(thresholds))
            return average_precision

# # debug code for above function
# confidences = [0.3, 0.9]
# predicted_boxes = [[20,20,60,70], [110,110,50,70]]
# target_boxes = [[25,25,60,70], [100,100,50,70]]#, [200, 200, 30, 50]]
# for box_p in predicted_boxes:
#     plt.imshow(box_mask(box_p, shape=256), cmap=mpl.cm.Reds, alpha=0.3)
# for box_t in target_boxes:
#     plt.imshow(box_mask(box_t, shape=256), cmap=mpl.cm.Greens, alpha=0.3)
# print(average_precision_image(predicted_boxes, confidences, target_boxes)) 

# define function that calculates the average precision of a batch of images
def average_precision_batch(output_batch, pIds, pId_boxes_dict, rescale_factor, shape=1024, return_array=False):
    """
    :param output_batch: cnn model output batch
    :param pIds: (list) list of patient IDs contained in the output batch
    :param rescale_factor: CNN image rescale factor
    :param shape: shape of the boolean masks (default set to maximum possible value, set to smaller to save memory)
    :returns: average_precision
    """
    
    batch_precisions = []
    for msk, pId in zip(output_batch, pIds): # iterate over batch prediction masks and relative patient IDs
        # retrieve target boxes from dictionary (quicker than from mask itself)
        target_boxes = pId_boxes_dict[pId] if pId in pId_boxes_dict else []
        # rescale coordinates of target boxes
        if len(target_boxes)>0:
            target_boxes = [[int(round(c/float(rescale_factor))) for c in box_t] for box_t in target_boxes]
        # extract prediction boxes and confidences
        predicted_boxes, confidences = parse_boxes(msk) 
        batch_precisions.append(average_precision_image(predicted_boxes, confidences, target_boxes, shape=shape))
    if return_array:
        return np.asarray(batch_precisions)
    else:
        return np.nanmean(np.asarray(batch_precisions)) 

# # debug code for above function
# targets = []
# pIds = []
# for i in range(5):
#     (img, target, pId) = dataset_train[i]
#     targets.append(target)
#     pIds.append(pId)
# # targets[0] = targets[1] #or pIds[0] = 'nan'
# average_precision_batch(targets, pIds, pId_boxes_dict, rescale_factor, shape=256)    


class RunningAverage():
    """A simple class that maintains the running average of a quantity
    
    Example:
    ```
    loss_avg = RunningAverage()
    loss_avg.update(2)
    loss_avg.update(4)
    loss_avg() = 3
    ```
    """
    def __init__(self):
        self.steps = 0
        self.total = 0
    
    def update(self, val):
        self.total += val
        self.steps += 1
    
    def __call__(self):
        return self.total/float(self.steps)



def save_checkpoint(state, is_best, metric):
    """Saves model and training parameters at 'last.pth.tar'. If is_best==True, also saves
    'best.pth.tar'
    Args:
        state: (dict) contains model's state_dict, may contain other keys such as epoch, optimizer state_dict
        is_best: (bool) True if it is the best model seen till now
    """
    filename = 'last.pth.tar'
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, metric+'.best.pth.tar')


# define the training function
def train(model, dataloader, optimizer, loss_fn, num_steps, pId_boxes_dict, rescale_factor, shape, save_summary_steps=5):
    # set model to train model
    model.train()
    
    summary = []
    loss_avg = RunningAverage()
    loss_avg_t_hist_ep, loss_t_hist_ep, prec_t_hist_ep = [], [], []

    # iterate over batches
    start = time.time()        
    
    for i, (input_batch, labels_batch, pIds_batch) in enumerate(dataloader):
        # break loop after num_steps batches (useful for debugging)
        if i > num_steps:
            break
        # Convert torch tensor to Variable
        input_batch = Variable(input_batch).cuda(async=True) if gpu_available else Variable(input_batch).float()
        labels_batch = Variable(labels_batch).cuda(async=True) if gpu_available else Variable(labels_batch).float()
            
        # compute output
        optimizer.zero_grad()
        output_batch = model(input_batch)

        # compute loss
        loss = loss_fn(output_batch, labels_batch)

        # compute gradient and do optimizer step
        loss.backward()
        optimizer.step()

        # update loss running average
        loss_avg.update(loss.item())
        loss_t_hist_ep.append(loss.item())
        loss_avg_t_hist_ep.append(loss_avg())

        # Evaluate summaries only once in a while
        if i % save_summary_steps == 0:
            # extract data from torch Variable, move to cpu
            output_batch = output_batch.data.cpu().numpy()
            # compute average precision on this batch
            prec_batch = average_precision_batch(output_batch, pIds_batch, pId_boxes_dict, rescale_factor, shape)
            prec_t_hist_ep.append(prec_batch)
            # log results
            summary_batch_string = "batch loss = {:05.7f} ;  ".format(loss.item())
            summary_batch_string += "average loss = {:05.7f} ;  ".format(loss_avg())
            summary_batch_string += "batch precision = {:05.7f} ;  ".format(prec_batch)
            print('--- Train batch {} / {}: '.format(i, num_steps) + summary_batch_string)
            delta_time = time.time() - start
            print('    {} batches processed in {:.2f} seconds'.format(save_summary_steps, delta_time))
            start = time.time()

    # log epoch summary
    metrics_string = "average loss = {:05.7f} ;  ".format(loss_avg())
    print("- Train epoch metrics summary: " + metrics_string)
    
    return loss_avg_t_hist_ep, loss_t_hist_ep, prec_t_hist_ep



def evaluate(model, dataloader, loss_fn, num_steps, pId_boxes_dict, rescale_factor, shape):
    # set model to evaluation mode
    model.eval()

    losses = []
    precisions = []

    # compute metrics over the dataset
    start = time.time()
    for i, (input_batch, labels_batch, pIds_batch) in enumerate(dataloader):
        # break loop after num_steps batches (useful for debugging)
        if i > num_steps:
            break
        # Convert torch tensor to Variable
        gpu_available = False
        input_batch = Variable(input_batch).cuda(async=True) if gpu_available else Variable(input_batch).float()
        labels_batch = Variable(labels_batch).cuda(async=True) if gpu_available else Variable(labels_batch).float()

        # compute model output
        output_batch = model(input_batch)
        # compute loss of batch
        loss = loss_fn(output_batch, labels_batch)
        losses.append(loss.item())

        # extract data from torch Variable, move to cpu
        output_batch = output_batch.data.cpu()
        # compute individual precisions of batch images
        prec_batch = average_precision_batch(output_batch, pIds_batch, pId_boxes_dict, rescale_factor, shape, return_array=True)
        for p in prec_batch:
            precisions.append(p)
        print('--- Validation batch {} / {}: '.format(i, num_steps))

    # compute mean of all metrics in summary
    metrics_mean = {'loss' : np.nanmean(losses),
                    'precision' : np.nanmean(np.asarray(precisions))}
    metrics_string = "average loss = {:05.7f} ;  ".format(metrics_mean['loss'])
    metrics_string += "average precision = {:05.7f} ;  ".format(metrics_mean['precision'])
    print("- Eval metrics : " + metrics_string)
    delta_time = time.time() - start
    print('  Evaluation run in {:.2f} seconds.'.format(delta_time))
    
    return metrics_mean


def train_and_evaluate(model, train_dataloader, val_dataloader, lr_init, loss_fn, num_epochs, 
                       num_steps_train, num_steps_eval, pId_boxes_dict, rescale_factor, shape, restore_file=None):

    # reload weights from restore_file if specified
    if restore_file is not None:
        checkpoint = torch.load(restore_file)
        model.load_state_dict(checkpoint['state_dict'])
            
    best_val_loss = 1e+15
    best_val_prec = 0.0
    best_loss_model = None
    best_prec_model = None

    loss_t_history = []
    loss_v_history = []
    loss_avg_t_history = []
    prec_t_history = []
    prec_v_history = []

    for epoch in range(num_epochs):
        start = time.time()
        
        # define the optimizer
        lr = lr_init * 0.5**float(epoch) # reduce the learning rate at each epoch
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        # Run one epoch
        print("Epoch {}/{}. Learning rate = {:05.3f}.".format(epoch + 1, num_epochs, lr))

        # train model for a whole epoc (one full pass over the training set)
        loss_avg_t_hist_ep, loss_t_hist_ep, prec_t_hist_ep = train(model, train_dataloader, optimizer, loss_fn, 
                                                                   num_steps_train, pId_boxes_dict, rescale_factor, shape)
        loss_avg_t_history += loss_avg_t_hist_ep
        loss_t_history += loss_t_hist_ep
        prec_t_history += prec_t_hist_ep
        
        # Evaluate for one epoch on validation set
        val_metrics = evaluate(model, val_dataloader, loss_fn, num_steps_eval, pId_boxes_dict, rescale_factor, shape)

        val_loss = val_metrics['loss']
        val_prec = val_metrics['precision']
        
        loss_v_history += len(loss_t_hist_ep) * [val_loss]
        prec_v_history += len(prec_t_hist_ep) * [val_prec]

        is_best_loss = val_loss<=best_val_loss
        is_best_prec = val_prec>=best_val_prec
        
        if is_best_loss:
            print("- Found new best loss: {:.4f}".format(val_loss))
            best_val_loss = val_loss
            best_loss_model = model
        if is_best_prec:
            print("- Found new best precision: {:.4f}".format(val_prec))
            best_val_prec = val_prec
            best_prec_model = model
            
        # Save best weights based on best_val_loss and best_val_prec
        save_checkpoint({'epoch': epoch + 1,
                         'state_dict': model.state_dict(),
                         'optim_dict' : optimizer.state_dict()},
                         is_best=is_best_loss,
                         metric='loss')
        save_checkpoint({'epoch': epoch + 1,
                         'state_dict': model.state_dict(),
                         'optim_dict' : optimizer.state_dict()},
                         is_best=is_best_prec,
                         metric='prec')
        
        delta_time = time.time() - start
        print('Epoch run in {:.2f} minutes'.format(delta_time/60.))

    histories = {'loss avg train' : loss_avg_t_history,
                 'loss train' : loss_t_history,
                 'precision train' : prec_t_history,
                 'loss validation' : loss_v_history, 
                 'precision validation' : prec_v_history}
    best_models = {'best loss model' : best_loss_model,
                   'best precision model' : best_prec_model}
    
    return histories, best_models




def predict(model, dataloader): 

    # set model to evaluation mode
    model.eval()
    
    predictions = {}

    for i, (test_batch, pIds) in enumerate(dataloader):
        print('Predicting batch {} / {}.'.format(i+1, len(dataloader)))
        # Convert torch tensor to Variable
        test_batch = Variable(test_batch).cuda(async=True) if gpu_available else Variable(test_batch).float()
            
        # compute output
        output_batch = model(test_batch)
        sig = nn.Sigmoid().cuda()
        output_batch = sig(output_batch)
        output_batch = output_batch.data.cpu().numpy()
        for pId, output in zip(pIds, output_batch):
            predictions[pId] = output
        
    return predictions



# train and evaluate the model
debug = False

# define an instance of the model
model = PneumoniaUNET().cuda() if gpu_available else PneumoniaUNET()
# define the loss function
loss_fn = BCEWithLogitsLoss2d().cuda() if gpu_available else BCEWithLogitsLoss2d()
# define initial learning rate (will be reduced over epochs)
lr_init = 0.5

num_epochs = 2 if debug else 2
num_steps_train = 50 if debug else len(loader_train)
num_steps_eval = 10 if debug else len(loader_valid)

shape = int(round(original_image_shape / rescale_factor))

# Train the model
print("Starting training for {} epochs".format(num_epochs))
histories, best_models = train_and_evaluate(model, loader_train, loader_valid, lr_init, loss_fn, 
                                            num_epochs, num_steps_train, num_steps_eval, pId_boxes_dict, rescale_factor, shape)


