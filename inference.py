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
# python3 inference.py PATH-TO-MAIN-FOLDER --weights PATH-TO-WEIGHTS
def testing_locanet():
    parser = argparse.ArgumentParser()
    parser.add_argument("foldername")
    parser.add_argument('--weights', type=str, default='/home/akmaral/tubCloud/Shared/cvmrs/trained-models/locanet/synth-5k', help='weights path(s)')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=(320,320), help='image size h,w')
    parser.add_argument('--stride', type=int, default=8, help='stride for depth,conf maps')
    parser.add_argument('--input_channel', type=int, default=1, help='image type RGB or GRAY')


    args = parser.parse_args()
    folder = str(Path(args.foldername))
    locanet_weights = args.weights
    input_size = args.imgsz
    input_channel = args.input_channel
    stride = args.stride
    shutil.rmtree(str(folder) + '/locanet/prediction/', ignore_errors=True)
    os.mkdir(str(folder) + '/locanet/prediction/')
    dist = 5
    testset = Dataset_Test(folder) 
    input_layer  = tf.keras.layers.Input([input_size[0], input_size[1], input_channel])

    feature_maps_locanet = locaNet(input_layer)
    model_locanet = tf.keras.Model(input_layer, feature_maps_locanet)
    model_locanet.load_weights(locanet_weights)
    start = perf_counter()
    predictions, images = {},{}
    for folder_image_name, image_data, target in testset:
        image_name = folder_image_name[0].split("/")[-1] 
        pred_neighbors, per_image = [], {}
        # predict with locanet
        pred_result_locanet = model_locanet(image_data, training=False)
        conf_locanet = tf.sigmoid(pred_result_locanet[0, :, :, 1:2]) 
        pos_conf_above_threshold_locanet = np.argwhere(conf_locanet > 0.33)
        list_pos_locanet = pos_conf_above_threshold_locanet.tolist()
        pos_conf_above_threshold_locanet = clean_array2d(list_pos_locanet, dist)
        img = cv2.imread(os.path.join(folder+'/Synchronized-Dataset/', folder_image_name[0]))  
        if (len(pos_conf_above_threshold_locanet) != 0):
            for j in range(len(pos_conf_above_threshold_locanet)): # for each predicted CF in image_i
                xy = pos_conf_above_threshold_locanet[j]
                curH = (xy[0]-0.5)*stride
                curW = (xy[1]+0.5)*stride   # get pixel values in image
                z_loca = (tf.exp(pred_result_locanet[0, xy[0], xy[1], 0])).numpy() # depth
                x_loca = z_loca*(curW-160)/170
                y_loca = z_loca*(curH-160)/170  # params used for data generation
                pred_neighbors.append(np.array([x_loca,y_loca,z_loca]))
                cv2.rectangle(img, (int(curW), int(curH)), (int(curW), int(curH)), (0, 0, 255), 4)
            if len(pred_neighbors):
                all_robots = {}
                for h in range(len(pred_neighbors)):
                    per_robot = {}
                    per_robot['pos'] = pred_neighbors[h].tolist() 
                    all_robots[h] = per_robot
                per_image['visible_neighbors'] = all_robots
                images[str(len(pred_neighbors)) + '/' + image_name] = per_image
                # visualize predictions
                cv2.rectangle(img, (int(curW), int(curH)), (int(curW), int(curH)), (0, 0, 255), 4)
                cv2.imwrite(os.path.join(folder+'/locanet/prediction/', image_name), img)
        else:
            per_image['visible_neighbors'] = []
            images[str(0) + '/' + image_name] = per_image
            cv2.imwrite(os.path.join(folder+'/locanet/prediction/', image_name), img)


    predictions['images'] = images
    with open(folder + '/locanet/inference_locanet.yaml', 'w') as outfile:
        yaml.dump(predictions, outfile)
    end = perf_counter()
    print("Time taken for test is {} min.".format((end-start)/60.))


if __name__ == "__main__":
    testing_locanet()