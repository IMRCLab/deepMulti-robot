import math
import numpy as np

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

