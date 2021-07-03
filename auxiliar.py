def collate_fn_prod(batch):
    images = []
    for i in batch:
        images.append(i)
    return images


def collate_fn(batch):
    images = []
    targets = []
    for i, t in batch:
        images.append(i)
        targets.append(t)
    return images, targets


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