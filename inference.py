import numpy as np
import tensorflow as tf
from time import perf_counter
from locaNet import  locaNet
from dataset_inference import Dataset_Test
import yaml
import shutil
import os
import cv2, argparse
import math
from pathlib import Path
# python3 inference.py PATH-TO-TEST-TXT --weights PATH-TO-WEIGHTS

def distance(a, b):
    return math.sqrt(pow((a[0]-b[0]), 2) + pow((a[1]-b[1]), 2))

def clean_array2d(array, threshold):
    i = 0
    while(i<len(array)):
        j = i+1
        while(j<len(array)):
            dist = distance(array[i], array[j])
            if dist < threshold:
                array.pop(j)
            else:
                j = j+1
        i = i+1
    return array


def inference(testfile, weights, imgsz=(320,320), stride=8, input_channel=1):

    test_txt = testfile
    locanet_weights = weights
    input_size = imgsz

    locanet_folder = Path(test_txt).parent
    prediction_folder = locanet_folder / "prediction"
    shutil.rmtree(prediction_folder, ignore_errors=True)
    os.mkdir(prediction_folder)

    dist = 5
    testset = Dataset_Test(test_txt) 
    input_layer  = tf.keras.layers.Input([input_size[0], input_size[1], input_channel])

    feature_maps_locanet = locaNet(input_layer)
    model_locanet = tf.keras.Model(input_layer, feature_maps_locanet)
    model_locanet.load_weights(locanet_weights)
    start = perf_counter()
    predictions, images = {},{}
    # read mapping
    with open(locanet_folder / "filename_to_dataset_mapping.yaml", 'r') as stream:
        filename_to_dataset_key = yaml.safe_load(stream)
    
    # read original data
    with open(Path(locanet_folder) / "dataset.yaml", 'r') as stream:
        filtered_dataset = yaml.safe_load(stream)

    for image_path, image_data, _ in testset:
        image_path = Path(image_path)
        image_name = str(image_path.name)
        key = str(image_path)
        pred_neighbors, per_image = [], {}
        # predict with locanet
        pred_result_locanet = model_locanet(image_data, training=False)
        conf_locanet = tf.sigmoid(pred_result_locanet[0, :, :, 1:2]) 
        pos_conf_above_threshold_locanet = np.argwhere(conf_locanet > 0.33)
        list_pos_locanet = pos_conf_above_threshold_locanet.tolist()
        pos_conf_above_threshold_locanet = clean_array2d(list_pos_locanet, dist)
        img = cv2.imread(str(image_path))  
        calibration_key = filtered_dataset['images'][filename_to_dataset_key[key]]['calibration']
        camera_matrix = np.array(filtered_dataset['calibration'][calibration_key]['camera_matrix'])
        dist_vec = np.array(filtered_dataset['calibration'][calibration_key]['dist_coeff'])
        fx,fy,ox,oy = camera_matrix[0][0], camera_matrix[1][1], camera_matrix[0][2], camera_matrix[1][2]
        if (len(pos_conf_above_threshold_locanet) != 0):
            for j in range(len(pos_conf_above_threshold_locanet)): # for each predicted CF in img
                xy = pos_conf_above_threshold_locanet[j]
                curH = (xy[0]-0.5)*stride
                curW = (xy[1]+0.5)*stride
                center = np.array([curH, curW], dtype=float)
                center_undistorted = cv2.undistortPoints(center, camera_matrix, dist_vec, None, camera_matrix).flatten()
                z_loca = (tf.exp(pred_result_locanet[0, xy[0], xy[1], 0])).numpy() 
                x_loca = z_loca*(center_undistorted[1]-ox)/fx
                y_loca = z_loca*(center_undistorted[0]-oy)/fy  
                pred_neighbors.append(np.array([x_loca,y_loca,z_loca]))
                # cv2.rectangle(img, (int(curW), int(curH)), (int(curW), int(curH)), (0, 0, 255), 4)
            if len(pred_neighbors):
                all_robots = []
                for h in range(len(pred_neighbors)):
                    per_robot = {}
                    per_robot['pos'] = pred_neighbors[h].tolist() 
                    all_robots.append(per_robot)
                per_image['visible_neighbors'] = all_robots
                images[filename_to_dataset_key[key]] = per_image
                # visualize predictions
                cv2.rectangle(img, (int(center_undistorted[1]), int(center_undistorted[0])), (int(center_undistorted[1]), int(center_undistorted[0])), (0, 0, 255), 4)
            cv2.imwrite(os.path.join(prediction_folder, image_name), img)
        else:
            per_image['visible_neighbors'] = []
            images[filename_to_dataset_key[key]] = per_image
            cv2.imwrite(os.path.join(prediction_folder, image_name), img)


    predictions['images'] = images

    end = perf_counter()
    print("Time taken for test is {} min.".format((end-start)/60.))

    with open(locanet_folder/"inference_locanet.yaml", 'w') as outfile:
        yaml.dump(predictions, outfile)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("testfile")
    parser.add_argument('--weights', type=str, default='/home/akmaral/tubCloud/Shared/cvmrs/trained-models/locanet/synth-5k', help='weights path(s)')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=(320,320), help='image size h,w')
    parser.add_argument('--stride', type=int, default=8, help='stride for depth,conf maps')
    parser.add_argument('--input_channel', type=int, default=1, help='image type RGB or GRAY')

    args = parser.parse_args()
    inference(args.testfile, args.weights, args.imgsz, args.stride, args.input_channel)

if __name__ == "__main__":
    main()