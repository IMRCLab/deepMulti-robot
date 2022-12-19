import numpy as np
import tensorflow as tf
from munkres import Munkres
from time import perf_counter

import deepMulti_robot.config as cfg
from deepMulti_robot.locaNet import  locaNet
from deepMulti_robot.dataset import Dataset
from deepMulti_robot.utils import *

def testing_locanet():
    Eucld_err, pr_loca, gt_loca =  [], [], []
    dist = 5
    m = Munkres()
    testset = Dataset('testing')  
    locanet_weights = '/home/akmaral/IMRS/cv-mrs/baselines/deepMulti_robot/output/' + cfg.OUTPUT_FILE
    input_size   = cfg.TRAIN_INPUT_SIZE
    input_layer  = tf.keras.layers.Input([input_size[0], input_size[1], cfg.INPUT_CHANNEL])

    feature_maps_locanet = locaNet(input_layer)
    model_locanet = tf.keras.Model(input_layer, feature_maps_locanet)
    model_locanet.load_weights(locanet_weights)
    start = perf_counter()
    for image_data, target in testset:
        # predict with locanet
        pred_result_locanet = model_locanet(image_data, training=False)
        conf_locanet = tf.sigmoid(pred_result_locanet[0, :, :, 1:2]) 
        pos_conf_above_threshold_locanet = np.argwhere(conf_locanet > 0.33)
        list_pos_locanet = pos_conf_above_threshold_locanet.tolist()
        pos_conf_above_threshold_locanet = clean_array2d(list_pos_locanet, dist)
        if (len(pos_conf_above_threshold_locanet) != 0):
            for j in range(len(pos_conf_above_threshold_locanet)): # for each predicted CF in image_i
                xy = pos_conf_above_threshold_locanet[j]
                curH = (xy[0]-0.5)*cfg.LOCA_STRIDE
                curW = (xy[1]+0.5)*cfg.LOCA_STRIDE   # get pixel values in image
                x_loca = (tf.exp(pred_result_locanet[0, xy[0], xy[1], 0])).numpy() # depth
                y_loca = -x_loca*(curW-160)/170
                z_loca = -x_loca*(curH-160)/170  # params used for data generation
                pr_loca.append(np.array([x_loca,y_loca,z_loca]))
            # get the ground truth for each single image
            xp, yp = np.where(target[0,:,:,0] != 0.)
            for k in range(len(xp)):             
                x_gt = float(target[:, xp[k], yp[k],0])/1000.0 # in m.
                y_gt = float(target[:, xp[k], yp[k],1])/1000.0
                z_gt = float(target[:, xp[k], yp[k],2])/1000.0
                gt_loca.append(np.array([x_gt, y_gt, z_gt]))
            if (len(pr_loca)!=0):
                matrix = find_euc(pr_loca, gt_loca) # matrix of Euclidean distance between all prediction vs. g-t
                _,cost = hungarian(m,matrix)
                Eucld_err.append(cost)
            pr_loca.clear()
            gt_loca.clear()
    end = perf_counter()
    print("Time taken for test is {} min.".format((end-start)/60.))
    # PlotWhisker.plot(Eucld_err, 'eculidean-synth-whisker-loca-mrs.jpg', "Box plot for multi-Cfs, synthetic data, locanet")
    # PlotWhisker.plot(Eucld_err, 'test.jpg', "test")
    return Eucld_err

if __name__ == "__main__":
    testing_locanet()