
import cv2
import shutil
import os
import yaml
import argparse
import glob
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
# Converts dataset.yaml into locanet training format. Images with no robot has just image-name
# python3 get_locanet_labels.py PATH-TO-MAIN-FOLDER
def run(main_data_folder, img_ext, train_data_percentage, data_split):
    synchronized_data_folder = main_data_folder + 'Synchronized-Dataset/'
    yaml_path = synchronized_data_folder + 'dataset.yaml'
    locanet_folder = main_data_folder + 'locanet/'
    shutil.rmtree(locanet_folder, ignore_errors=True)
    os.mkdir(locanet_folder) # Create a folder for saving images
    
    with open(yaml_path, 'r') as stream:
        synchronized_data = yaml.safe_load(stream)
    files = ['train', 'val']   
    # for folder in [f.path for f in os.scandir(synchronized_data_folder) if f.is_dir()]:
    for key in data_split: # for each folder 0,1,2
        total_imgs = sorted(filter(os.path.isfile, glob.glob(synchronized_data_folder + key + '/' + img_ext)))
        if (len(total_imgs)) or data_split[key] <= len(total_imgs):
            indices = list(range(0, data_split[key])) 
            numImgTrain = round(train_data_percentage/100*len(indices))
            training_idx, val_idx = indices[:numImgTrain], indices[numImgTrain:]
            idx = [training_idx,val_idx]
            for k in range(len(idx)): # train, val
                file_name = locanet_folder + files[k] + '.txt'
                fileTmp = open(file_name, 'a')
                for t in idx[k]: # for each image
                    dataLine = key + '/' + total_imgs[t].split("/")[-1] + ' ' # 1/img_name
                    for j in range(len(synchronized_data['images'][key + '/' + total_imgs[t].split("/")[-1]]['visible_neighbors'])):
                        pix = synchronized_data['images'][key + '/' + total_imgs[t].split("/")[-1]]['visible_neighbors'][j]['pix']
                        pos = synchronized_data['images'][key + '/' + total_imgs[t].split("/")[-1]]['visible_neighbors'][j]['pos']
                        dataLine += str(int(pix[0])) + ',' + str(int(pix[1])) + ',' + str(round(pos[0]*1000)) + ',' + str(round(pos[1]*1000)) + ',' + str(round(pos[2]*1000)) + ' ' # m->mm
                    dataLine += '\n'
                    fileTmp.write(dataLine)
        else:
            print('Not Enough Images in folder {}'.format(key))
            return 

    plt.title('Training Dataset')
    plt.bar(list(data_split.keys()), data_split.values(), color='g')
    plt.ylabel('Number of Images')
    plt.xlabel('Number of Robots')
    plt.savefig(main_data_folder + 'train_robot_number.jpg')
    
def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('foldername', help="Path to the Synchronized-Dataset folder")
    parser.add_argument('--img_ext', type=str, default= '*.jpg', help="image extension") 
    parser.add_argument('--training_data_percentage', type=int, default=100, help='training data percentage')
    parser.add_argument('--data_split', default={'0': 180, '1':950, '2': 1300}, help='percentage for data split between different number of robots')

    args = parser.parse_args()
    return args

def main(args):
    run(args.foldername, args.img_ext, args.training_data_percentage, args.data_split)
    


if __name__ == "__main__":
    args = parse_opt()
    main(args)