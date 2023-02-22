import yaml
import argparse
from pathlib import Path
import itertools
import os

# python3 get_locanet_labels.py -f arg1 -f arg2 
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f','--file', type=str, nargs='+', action='append', help='file list')
    parser.add_argument('-mode',help='training or test')
    args = parser.parse_args()
    data = args.file  
    mode = args.mode

    folder = Path(data[0][0]).parent.parent.parent / "locanet"
    os.mkdir(folder)
    if mode == 'train':
        file_name = str(folder) + '/train.txt'      
    else:
        file_name = str(folder) + '/test.txt' 
    fileTmp = open(file_name, 'a')
    for i in range(len(data)): 
        main_folder =  str(Path(data[i][0]).parent) 
        yaml_file = data[i][0]
        with open(yaml_file, 'r') as stream:
            synchronized_data = yaml.safe_load(stream)
        training_data = dict(itertools.islice(synchronized_data['images'].items(), int(data[i][1])))
        for image, value in training_data.items():
            dataLine = main_folder + '/' +  image + ' ' 
            neighbors = value['visible_neighbors']
            if len(neighbors) > 0:
                for neighbor in neighbors:
                    cx, cy = neighbor['pix']
                    x, y, z = neighbor['pos']
                    if cx >= 320 or cy >= 320 or cx <= 0 or cy <= 0: 
                            continue
                    dataLine += str(int(cx)) + ',' + str(int(cy)) + ',' + str(round(x*1000)) + ',' + str(round(y*1000)) + ',' + str(round(z*1000)) + ' ' # m->mm
                dataLine += '\n'
            else:
                dataLine += '\n'
            fileTmp.write(dataLine)
        

if __name__ == "__main__":
    main()

