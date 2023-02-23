import yaml
import argparse
from pathlib import Path
import itertools
import os
import shutil
# python3 get_locanet_labels.py ptha-to-yaml-file
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f','--file', type=str, help='dataset.yaml file')
    parser.add_argument('-mode',help='training or test')
    args = parser.parse_args()
    data = args.file  
    mode = args.mode

    main_folder =  Path(args.file).parent.parent

    locanet_folder = main_folder / "locanet"
    shutil.rmtree(locanet_folder, ignore_errors=True)
    os.mkdir(locanet_folder) # Create a folder for saving images

    if mode == 'train':
        file_name = locanet_folder / 'train.txt'      
    else:
        file_name = locanet_folder / 'test.txt' 

    fileTmp = open(file_name, 'a')
    yaml_file = args.file
    with open(yaml_file, 'r') as stream:
        synchronized_data = yaml.safe_load(stream)

    filename_to_dataset_key = dict()
    for image_name, entry in synchronized_data['images'].items():

        new_image_name = str(Path(image_name).name)
        if new_image_name in filename_to_dataset_key:
            # already there -> resolve duplicates
            i = 0
            while True:
                new_image_name = str(Path(image_name).stem) + "_" + str(i) + str(Path(image_name).suffix)
                if new_image_name not in filename_to_dataset_key:
                    break
        filename_to_dataset_key[new_image_name] = image_name
        neighbors = entry['visible_neighbors']
        dataLine = str(Path(args.file).parent / image_name)
        for neighbor in neighbors:
            cx, cy = neighbor['pix']
            x, y, z = neighbor['pos']
            # if cx >= 320 or cy >= 320 or cx <= 0 or cy <= 0: 
            #     continue
            dataLine += str(int(cx)) + ',' + str(int(cy)) + ',' + str(round(x*1000)) + ',' + str(round(y*1000)) + ',' + str(round(z*1000)) + ' ' # m->mm
        dataLine += '\n'
        fileTmp.write(dataLine)
        
if __name__ == "__main__":
    main()

