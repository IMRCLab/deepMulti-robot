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

def get_euclidean_err(a,b): # input: two arrays of x,y,z
    return math.sqrt(pow((a[0] - b[0]), 2) + pow((a[1] - b[1]), 2) + pow((a[2] - b[2]), 2))

def find_min_euc(pr_list, gt_list):
    f = []
    for i in range(len(pr_list)):
        min_err = 10.
        gt_index = -1
        for j in gt_list:
            temp_err = get_euclidean_err(pr_list[i], j)
            if min_err > temp_err:
                min_err=temp_err
                gt_index += 1 # 0 for the first element
        f.append(min_err) # Euclidean error for each prediction vs. g.t
        del gt_list[gt_index]
        
        if not gt_list:
            break
    return f
    
# get Euclidean distance for all prediction with each possible g-t.
def find_euc(pr_list, gt_list):
    matrix = []
    for i in range(len(gt_list)):
        tmp_list = []
        for j in pr_list:
            tmp_list.append(get_euclidean_err(gt_list[i], j))
        matrix.append(tmp_list)
    return matrix

# returns indices for each row and total sum of cost
def hungarian(m,mx):
    indexes = m.compute(mx)
    total = 0
    for row, column in indexes:
        value = mx[row][column]
        total += value
    return indexes, total