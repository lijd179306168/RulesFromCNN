# -*- coding: utf-8 -*-
import numpy as np
import csv

s= [0, 1, 22, 96, 110, 191, 231, 239, 242, 267, 317, 384, 398, 435]#activatoinsValue>=0.45 and activatoinsCount>=7000

x = np.loadtxt("cifar10train_50000_512.csv", skiprows=0,delimiter=",",usecols=s)

with open('cifar10train_50000_12.csv','w',newline='')as f:
    f_csv = csv.writer(f)
    for i in range(len(x)):
        f_csv.writerow(x[i])
        
print("ok")
