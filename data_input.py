import pandas as pd
import numpy
import csv
import os 
from glob import glob


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


def initial_dataframe():
        det_class_path = '/home/medicine_project/input_data/stage_2_detailed_class_info.csv'
        bbox_path = '/home/medicine_project/input_data/stage_2_train_labels.csv'
        dicom_dir = '/home/medicine_project/input_data/stage_2_train_images/'
        det_class_df = pd.read_csv(det_class_path)
        bbox_df = pd.read_csv(bbox_path)
        comb_box_df = pd.concat([bbox_df, det_class_df.drop('patientId',1)], 1)

        image_df = pd.DataFrame({'path': glob(os.path.join(dicom_dir, '*.dcm'))})
        image_df['patientId'] = image_df['path'].map(lambda x: os.path.splitext(os.path.basename(x))[0])
        print(image_df.shape[0], 'images found')
        img_pat_ids = set(image_df['patientId'].values.tolist())
        box_pat_ids = set(comb_box_df['patientId'].values.tolist())
        # check to make sure there is no funny business
        #assert img_pat_ids.union(box_pat_ids)==img_pat_ids, "Patient IDs should be the same"
        image_bbox_df = pd.merge(comb_box_df, 
                         image_df, 
                         on='patientId',
                        how='left').sort_values('patientId')
        print(image_bbox_df.shape[0], 'image bounding boxes')
        return(image_bbox_df)


#dd = initial_dataframe()
#print(dd.head(n =10))
#print(dd.columns)
#print(dd.shape)

