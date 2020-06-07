# -*- coding: utf-8 -*-
import numpy as np
import csv

activatoinsValue=0.45
activatoinsCount=7000

with open('cifar10train_50000_512.csv','r')as f:
    f_csv = csv.reader(f)
    scount=dict(zip([i for i in range(514)],[0 for i in range(514)]))
    print("scount=",scount)
    for row in f_csv:
        tmpx=row[:]
        print(tmpx)
        for i in range(len(tmpx)):
            if float(tmpx[i])>=activatoinsValue:
                scount[i]=scount[i]+1

    print("scount=",scount)
    s=set()
    for i in range(len(scount)):
        if  scount[i]>activatoinsCount:
            s.add(i)   
    s=sorted(s)
    print("len s=",len(s))
    print("s=",s)
     
print('finish')