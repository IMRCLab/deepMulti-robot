import numpy as np
import tensorflow as tf
from time import perf_counter
import config as cfg
from locaNet import  locaNet
from dataset_inference import Dataset_Test
from utils import *
import yaml
import shutil
import os
import cv2, argparse
from pathlib import Path
# Tested with single Cf case
def testing_locanet():
    parser = argparse.ArgumentParser()
    parser.add_argument("foldername")
    args = parser.parse_args()
    folder = Path(args.foldername)
     
    shutil.rmtree(str(folder) + '/locanet/prediction/', ignore_errors=True)
    os.mkdir(str(folder) + '/locanet/prediction/')
    dist = 5
    testset = Dataset_Test(cfg) 
    locanet_weights = cfg.WEIGHT_PATH + cfg.OUTPUT_FILE
    input_size   = cfg.TRAIN_INPUT_SIZE
    input_layer  = tf.keras.layers.Input([input_size[0], input_size[1], cfg.INPUT_CHANNEL])

    feature_maps_locanet = locaNet(input_layer)
    model_locanet = tf.keras.Model(input_layer, feature_maps_locanet)
    model_locanet.load_weights(locanet_weights)
    start = perf_counter()
    predictions, images = {},{}
    for image_name, image_data, target in testset:
        pred_neighbors, per_image = [], {}
        # predict with locanet
        pred_result_locanet = model_locanet(image_data, training=False)
        conf_locanet = tf.sigmoid(pred_result_locanet[0, :, :, 1:2]) 
        pos_conf_above_threshold_locanet = np.argwhere(conf_locanet > 0.33)
        list_pos_locanet = pos_conf_above_threshold_locanet.tolist()
        pos_conf_above_threshold_locanet = clean_array2d(list_pos_locanet, dist)
        if (len(pos_conf_above_threshold_locanet) != 0):
            img = cv2.imread(os.path.join(cfg.DATASET_FOLDER, image_name[0]))  
            for j in range(len(pos_conf_above_threshold_locanet)): # for each predicted CF in image_i
                xy = pos_conf_above_threshold_locanet[j]
                curH = (xy[0]-0.5)*cfg.LOCA_STRIDE
                curW = (xy[1]+0.5)*cfg.LOCA_STRIDE   # get pixel values in image
                x_loca = (tf.exp(pred_result_locanet[0, xy[0], xy[1], 0])).numpy() # depth
                y_loca = -x_loca*(curW-160)/170
                z_loca = -x_loca*(curH-160)/170  # params used for data generation
                pred_neighbors.append(np.array([x_loca,y_loca,z_loca]))
            if len(pred_neighbors):
                # shutil.copy(path + robot_name[i] + '/' + img_names[t], synch_data_path + img_names[t])
                all_robots = {}
                for h in range(len(pred_neighbors)):
                    per_robot = {}
                    per_robot['pos'] = pred_neighbors[h].tolist() 
                    all_robots[h] = per_robot
                per_image['visible_neighbors'] = all_robots
                images[image_name[0]] = per_image
                # visualize predictions
                cv2.rectangle(img, (int(curW), int(curH)), (int(curW), int(curH)), (0, 0, 255), 4)
                cv2.imwrite(os.path.join(cfg.DATASET_FOLDER+ '../locanet/prediction/', image_name[0]), img)

    predictions['images'] = images
    with open(cfg.INFERENCE_FILE, 'w') as outfile:
        yaml.dump(predictions, outfile)
    end = perf_counter()
    print("Time taken for test is {} min.".format((end-start)/60.))


if __name__ == "__main__":
    testing_locanet()