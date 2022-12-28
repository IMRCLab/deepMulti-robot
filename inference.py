import numpy as np
import tensorflow as tf
from time import perf_counter
import config as cfg
from locaNet import  locaNet
from dataset_inference import Dataset_Test
from utils import *
import yaml
# Tested with single Cf case
def testing_locanet():
    xyz_loca = []
    prediction={}
    dist = 5
    testset = Dataset_Test() 
    locanet_weights = cfg.WEIGHT_PATH + cfg.OUTPUT_FILE
    input_size   = cfg.TRAIN_INPUT_SIZE
    input_layer  = tf.keras.layers.Input([input_size[0], input_size[1], cfg.INPUT_CHANNEL])

    feature_maps_locanet = locaNet(input_layer)
    model_locanet = tf.keras.Model(input_layer, feature_maps_locanet)
    model_locanet.load_weights(locanet_weights)
    start = perf_counter()
    success = 0
    for image_name, image_data, target in testset:
        # predict with locanet
        pred_result_locanet = model_locanet(image_data, training=False)
        conf_locanet = tf.sigmoid(pred_result_locanet[0, :, :, 1:2]) 
        pos_conf_above_threshold_locanet = np.argwhere(conf_locanet > 0.33)
        list_pos_locanet = pos_conf_above_threshold_locanet.tolist()
        pos_conf_above_threshold_locanet = clean_array2d(list_pos_locanet, dist)
        if (len(pos_conf_above_threshold_locanet) != 0):
            success += 1
            for j in range(len(pos_conf_above_threshold_locanet)): # for each predicted CF in image_i
                xy = pos_conf_above_threshold_locanet[j]
                curH = (xy[0]-0.5)*cfg.LOCA_STRIDE
                curW = (xy[1]+0.5)*cfg.LOCA_STRIDE   # get pixel values in image
                x_loca = (tf.exp(pred_result_locanet[0, xy[0], xy[1], 0])).numpy() # depth
                y_loca = -x_loca*(curW-160)/170
                z_loca = -x_loca*(curH-160)/170  # params used for data generation
                xyz_loca.append(np.array((x_loca,y_loca,z_loca)).tolist())
            prediction[str(image_name[0])] = xyz_loca[:]
            del xyz_loca[:] 
        else:
            prediction[str(image_name[0])] = np.array((None,None,None)).tolist()

    with open(cfg.INFERENCE_FILE, 'w') as outfile:
        yaml.dump(prediction, outfile)
    end = perf_counter()
    print("Time taken for test is {} min.".format((end-start)/60.))
    print("Success rate is {} for {} images.".format(success*100/len(testset), len(testset)))

if __name__ == "__main__":
    testing_locanet()