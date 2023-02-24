import yaml
import argparse
from pathlib import Path
import itertools
import os
import shutil
# python3 get_locanet_labels.py ptha-to-yaml-file

def get_labels(file, output_folder, mode):
    locanet_folder = Path(output_folder) / "locanet"
    shutil.rmtree(locanet_folder, ignore_errors=True)
    os.mkdir(locanet_folder) # Create a folder for saving images

    if mode == 'train':
        file_name = locanet_folder / 'train.txt'      
    else:
        file_name = locanet_folder / 'test.txt' 

    fileTmp = open(file_name, 'a')
    yaml_file = file
    with open(yaml_file, 'r') as stream:
        synchronized_data = yaml.safe_load(stream)

    filename_to_dataset_key = dict()
    for image_name, entry in synchronized_data['images'].items():
        neighbors = entry['visible_neighbors']
        dataLine = str((Path(file).parent / image_name).absolute())
        filename_to_dataset_key[dataLine] = image_name
        for neighbor in neighbors:
            cx, cy = neighbor['pix']
            x, y, z = neighbor['pos']
            # if cx >= 320 or cy >= 320 or cx <= 0 or cy <= 0: 
            #     continue
            dataLine += ' ' + str(int(cx)) + ',' + str(int(cy)) + ',' + str(round(x*1000)) + ',' + str(round(y*1000)) + ',' + str(round(z*1000)) + ' ' # m->mm
        dataLine += '\n'
        fileTmp.write(dataLine)

    with open(locanet_folder / "filename_to_dataset_mapping.yaml", "w") as f:
        yaml.dump(filename_to_dataset_key, f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f','--file', type=str, help='dataset.yaml file')
    parser.add_argument('-mode',help='training or test')
    args = parser.parse_args()

    main_folder =  Path(args.file).parent.parent

    get_labels(args.file, main_folder, args.mode)

if __name__ == "__main__":
    main()
