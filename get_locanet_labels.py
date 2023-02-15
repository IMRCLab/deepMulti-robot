
import cv2
import shutil
import os
import yaml
import argparse
import glob
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from os.path import normpath, basename
# Converts dataset.yaml into locanet training format. Images with no robot has just image-name
# python3 get_locanet_labels.py PATH-TO-MAIN-FOLDER
def run(main_data_folder, img_ext, train_data_percentage):

    synchronized_data_folder = main_data_folder + 'Synchronized-Dataset/'
    yaml_path = synchronized_data_folder + 'dataset.yaml'
    locanet_folder = main_data_folder + 'locanet/'
    shutil.rmtree(locanet_folder, ignore_errors=True)
    os.mkdir(locanet_folder) # Create a folder for saving images
    
    with open(yaml_path, 'r') as stream:
        synchronized_data = yaml.safe_load(stream)
    files = ['train', 'val', 'test']   
    train_robot_number, test_robot_number = np.zeros(6), np.zeros(6)
    for folder in [f.path for f in os.scandir(synchronized_data_folder) if f.is_dir()]:
        robot_number = basename(normpath(folder)) # get the folder with robot number
        total_imgs = sorted(filter( os.path.isfile, glob.glob(folder + '/' + img_ext) ) )
        # Prepare training, validation, testing data
        indices = list(range(0, len(total_imgs))) 
        numImgTrain = round(train_data_percentage/100*len(total_imgs))
        training_idx, val_idx, test_idx = indices[:numImgTrain], indices[numImgTrain:numImgTrain+int((len(total_imgs)-numImgTrain)/2)], indices[numImgTrain+int((len(total_imgs)-numImgTrain)/2):]
        train_robot_number[int(robot_number)] += len(training_idx)
        test_robot_number[int(robot_number)] += len(test_idx)
        idx = [training_idx,val_idx,test_idx]
        for k in range(len(idx)): # train, val, test
            file_name = locanet_folder + files[k] + '.txt'
            fileTmp = open(file_name, 'a')
            for t in idx[k]: # for each image
                dataLine = str(robot_number) + '/' + total_imgs[t].split("/")[-1] + ' ' # 1/img_name
                for j in range(len(synchronized_data['images'][str(robot_number) + '/' + total_imgs[t].split("/")[-1]]['visible_neighbors'])):
                    pix = synchronized_data['images'][str(robot_number) + '/' + total_imgs[t].split("/")[-1]]['visible_neighbors'][j]['pix']
                    pos = synchronized_data['images'][str(robot_number) + '/' + total_imgs[t].split("/")[-1]]['visible_neighbors'][j]['pos']
                    dataLine += str(int(pix[0])) + ',' + str(int(pix[1])) + ',' + str(round(pos[0]*1000)) + ',' + str(round(pos[1]*1000)) + ',' + str(round(pos[2]*1000)) + ' ' # m->mm
                dataLine += '\n'
                fileTmp.write(dataLine)

    # Training, Testing number of robots
    fig, axs = plt.subplots(1, 2, sharey=True, sharex=True)
    fig.suptitle('Used Number of Robots')
    axs[0].set_title("Training", fontsize=10)
    axs[0].bar([*range(0, len(train_robot_number.tolist()), 1)], train_robot_number.tolist())
    axs[1].set_title("Testing", fontsize=10)
    axs[1].bar([*range(0, len(test_robot_number.tolist()), 1)], test_robot_number.tolist())
    fig.text(0.5, 0.04, 'Number of Images', va='center', ha='center')
    fig.text(0.04, 0.5, 'Number of Robots', va='center', ha='center', rotation='vertical')
    plt.savefig(main_data_folder + 'train_test_robot_number.jpg')
    
        
def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('foldername', help="Path to the Synchronized-Dataset folder")
    parser.add_argument('--img_ext', type=str, default= '*.jpg', help="image extension") 
    parser.add_argument('--training_data', type=int, default=90, help='training data percentage')


    args = parser.parse_args()
    return args

def main(args):
    run(args.foldername, args.img_ext, args.training_data)


if __name__ == "__main__":
    args = parse_opt()
    main(args)