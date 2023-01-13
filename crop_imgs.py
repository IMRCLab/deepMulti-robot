# crop pixels from both sides 324x324 to get 320x320 for locanet
import numpy as np
import cv2
import random
import os
import config as cfg
import shutil
import argparse
# Real images have 324x324 size, crop it to fit NN input.
parser = argparse.ArgumentParser(description='Connect to AI-deck JPEG streamer example')

parser.add_argument("-path", help="Path to the folder with 324x324 size images")
args = parser.parse_args()

y=2
x=2
h=320
w=320

full_data_path = args.path
cropped_images_path = full_data_path + '../Synchronized-Dataset-2/'
shutil.rmtree(cropped_images_path, ignore_errors=True)
os.mkdir(cropped_images_path)
extension_allowed = '.png'

ext_len = len(extension_allowed)
files = []
for r, d, f in os.walk(full_data_path):
    for file in f:
        if file.endswith(extension_allowed):
            strip = file[0:len(file) - ext_len] 
            files.append(strip)
random.shuffle(files)
size = len(files)                   
# split = int(split_percentage * size / 100)
print("starting to crop images")
for i in range(size):
    strip = files[i]
    image_file = strip + extension_allowed
    src_image = full_data_path  + image_file
    image = cv2.imread(src_image)
    crop = image[y:y+h, x:x+w]
    cv2.imwrite(os.path.join(cropped_images_path, image_file), crop)
#     shutil.copy(crop, cropped_images_path) 
                         
