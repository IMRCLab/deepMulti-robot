
import cv2
import shutil
import os
import yaml
import argparse
import glob
# Converts dataset.yaml into locanet training format. Images with no robot has just image-name
# python3 get_locanet_labels.py PATH-TO-SYNCHRONIZED-DATASET/
def run(synchronized_data_folder, img_ext, train_data_percentage):

    yaml_path = synchronized_data_folder + 'dataset.yaml'
    locanet_folder = synchronized_data_folder + '../locanet/'
    shutil.rmtree(locanet_folder, ignore_errors=True)
    os.mkdir(locanet_folder) # Create a folder for saving images
    
    with open(yaml_path, 'r') as stream:
        synchronized_data = yaml.safe_load(stream)
    # total_imgs = fnmatch.filter(os.listdir(synchronized_data_folder), img_ext)
    total_imgs = sorted(filter( os.path.isfile, glob.glob(synchronized_data_folder + img_ext) ) )
    # Prepare training, validation, testing data
    indices = list(range(0, len(total_imgs))) 
    numImgTrain = round(train_data_percentage/100*len(total_imgs))
    training_idx, val_idx, test_idx = indices[:numImgTrain], indices[numImgTrain:numImgTrain+int((len(total_imgs)-numImgTrain)/2)], indices[numImgTrain+int((len(total_imgs)-numImgTrain)/2):]
    idx = [training_idx,val_idx,test_idx]
    files = ['train', 'val', 'test']    
    for k in range(len(idx)): 
        file_name = locanet_folder + files[k] + '.txt'
        fileTmp = open(file_name, 'w')
        for t in idx[k]: # for each image
            dataLine = total_imgs[t].split("/")[-1] + ' ' #total_imgs[t][-13:] + ' '
            for j in range(len(synchronized_data['images'][total_imgs[t].split("/")[-1]]['visible_neighbors'])):
                pix = synchronized_data['images'][total_imgs[t].split("/")[-1]]['visible_neighbors'][j]['pix']
                pos = synchronized_data['images'][total_imgs[t].split("/")[-1]]['visible_neighbors'][j]['pos']
                dataLine += str(int(pix[0])) + ',' + str(int(pix[1])) + ',' + str(round(pos[0]*1000)) + ',' + str(round(pos[1]*1000)) + ',' + str(round(pos[2]*1000)) + ' ' # m->mm
            dataLine += '\n'
            fileTmp.write(dataLine)
        
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