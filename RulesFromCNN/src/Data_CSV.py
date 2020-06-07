# -*- coding: utf-8 -*-
import numpy as np
import csv
from sklearn import preprocessing  

class LoadDataCSV(object):
    def __init__(self):        
        self.dataset=[]
        self.normaldataset=[]
        self.trainset=[]
        self.trainsetClassAmount=[]
        self.validateset=[]
        self.validatesetClassAmount=[]
        self.classLabel=['0','1','2','3','4','5','6','7','8','9']        
        with open('cifar10train_50000_12.csv','r')as f:
            f_csv = csv.reader(f)
            print(f_csv)
            tmpdatax=[]
            for row in f_csv:
                tmpx=row[2:]
                tmpy=row[1:2]
                tmpid=row[0:1]
                x=[]
                y=[]
                id=[]
                for j in range(len(tmpx)):
                    x.append(float(tmpx[j]))
                for j in range(len(tmpy)):
                    y.append(tmpy[j])
                for j in range(len(tmpid)):
                    id.append(int(tmpid[j]))
                tmpdatax.append(x)
                self.dataset.append([x,y,id])
            
        tmpdatax=self.normalization_data(tmpdatax)       
        self.normaldataset=[]
        
        for i in range(len(self.dataset)):
            x=[]
            y=[]
            id=[]
            for j in range(len(tmpdatax[i])):
                x.append(float(tmpdatax[i][j]))
            y.append(self.dataset[i][1][0])
            id.append(self.dataset[i][2][0])
            self.normaldataset.append([x,y,id])
        np.random.shuffle(self.normaldataset)
        totalamount=len(self.normaldataset)
        print("totalamount",totalamount)
        trainamount=int(99*totalamount/100)
        print("trainamount",trainamount)
        self.trainset=self.normaldataset[:trainamount]
        self.validateset=self.normaldataset[trainamount: ]
        print('load ok')
    def normalization_data(self,datax):
        min_max_scaler = preprocessing.MinMaxScaler() 
        X_minMax = min_max_scaler.fit_transform(datax)
        return X_minMax
    def get_classLabel(self):
        return self.classLabel            
    def get_train_data(self):
        self.trainsetClassAmount=dict(zip(self.classLabel,[0 for i in range(len(self.classLabel))]))
        for eachexample in self.trainset:
            y=eachexample[1][0]
            self.trainsetClassAmount[y]=self.trainsetClassAmount[y]+1
        return self.trainset,self.trainsetClassAmount
    def get_validate_data(self):    
        self.validatesetClassAmount=dict(zip(self.classLabel,[0 for i in range(len(self.classLabel))]))   
        for eachexample in self.validateset:
            y=eachexample[1][0]
            self.validatesetClassAmount[y]=self.validatesetClassAmount[y]+1
        return self.validateset,self.validatesetClassAmount
    def get_data_batch(self,bfrom,bto):
        x=[]
        y=[]
        for item in self.trainset[bfrom:bto]:
            tempx=np.asarray(item[0])
            x.append(tempx)
            y.append(item[1])
        return np.asarray(x), np.asarray(y)

    
