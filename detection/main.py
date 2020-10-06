import os
import numpy as np
import csv

def read_csv_to_numpy(data_path): 
    datas = csv.reader(open(data_path,'r'))
    temp = []
    for data in datas:
        temp.append(data)
    temp = temp[1:]
    #print(len(temp))
    data_numpy = np.zeros((len(temp)-1,3))
    for i in range(len(temp)-1):
        data_numpy[i,:] = np.array([int(temp[i][0]),float(temp[i][1]),int(temp[i][2])])
    return data_numpy


def find_change_point(data,threshold): 
    bias = data[:, 1]
    change_index = []
    
    for i in range(len(bias) - 1):
        if bias[i] >= threshold:
            change_index.append(i)
    return change_index

def culculate_score(data,change_index): 
    n=0
    true_class = data[:,2]
    for p in change_index:
        if true_class[p] ==1:
            n+=1
    precision = n/len(change_index)

    total_change_point=0
    for m in true_class:
        if m ==1:
            total_change_point+=1
    recall = n/total_change_point
    F1_score = 2*precision*recall/(precision+recall)

    return precision,recall,F1_score


def detect_change_point_start_location(data,change_index):
    t1=[]
    t2 = [] 
    for x in change_index:
        t1.append(x)
        if x+1 not in change_index:
            t2.append([t1[0],t1[-1]])
            t1=[]
        else:
            continue

    change_point_start_location = []
    for i in range(len(t2)-1):
        if t2[i][1]-t2[i][0]+1 >=3:
            change_point_start_location.append(t2[i][0])

    return change_point_start_location


if __name__ == '__main__':
    data_path = '/Users/duanzy/Desktop/code1/1-detection/2.air.csv' 
    threshold = 8  
    data = read_csv_to_numpy(data_path) 
    change_point_index = find_change_point(data,threshold)
    precision,recall,F1_score=culculate_score(data,change_point_index)
    change_point_start_location=detect_change_point_start_location(data,change_point_index)
    print("precision=", precision)
    print("recall = ", recall)
    print("F1_score=", F1_score)
    print('变化点位置：',change_point_start_location)

