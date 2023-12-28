import argparse
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pygad
# import wandb
import pandas as pd
import tensorflow.keras.backend as K
from tensorflow.python.client import device_lib
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

# print(device_lib.list_local_devices())

import matplotlib.pyplot as plt
tf.random.set_seed(0)
np.random.seed(0)



def real_world_csv_data_loader(name):

    if name == "10-speech_pca.csv":

        csv_file = './data/10-speech_pca.csv'
        label_csv_files = ['./data_od_evaluation/speech_pca_gt_copod.csv', 
                           './data_od_evaluation/speech_pca_gt_hbos.csv',
                           './data_od_evaluation/speech_pca_gt_iforest.csv']
        data = pd.read_csv(csv_file)

        gt_copod = pd.read_csv(label_csv_files[0])
        gt_hbos = pd.read_csv(label_csv_files[1])
        gt_iforest = pd.read_csv(label_csv_files[2])

        last_column = data.iloc[:, -1].values
        # last_column = np.where(last_column == 0, 0, 1)
        labels = last_column
        data = np.array(data)
        GTs = [gt_copod, gt_hbos, gt_iforest]
        data_n = data[:,:-1]
    
    
    if name == "09-satimage-2_pca.csv":

        csv_file = './data/09-satimage-2_pca.csv'
        label_csv_files = ['./data_od_evaluation/satimage-2_pca_gt_copod.csv', 
                           './data_od_evaluation/satimage-2_pca_gt_hbos.csv',
                           './data_od_evaluation/satimage-2_pca_gt_iforest.csv']
        data = pd.read_csv(csv_file)

        gt_copod = pd.read_csv(label_csv_files[0])
        gt_hbos = pd.read_csv(label_csv_files[1])
        gt_iforest = pd.read_csv(label_csv_files[2])

        last_column = data.iloc[:, -1].values
        # last_column = np.where(last_column == 0, 0, 1)
        labels = last_column
        data = np.array(data)
        GTs = [gt_copod, gt_hbos, gt_iforest]
        data_n = data[:,:-1]

    

    return data_n, labels, GTs