#coding=utf-8
import numpy as np
import random


class Triangle(object):
    def __init__(self):  
        
        self.shapes={'s': [-0.01,0,0.25], 'ls': [-0.01,0,0.4], 'mls': [-0.01,0,0.33333],
                     'vs': [-0.01,0,0.1667], 'es': [-0.01,0,0.1],
                     'ms': [0,0.25,0.5], 'lms': [-0.15,0.25,0.65], 'mlms': [-0.0833,0.25,0.5833],
                     'vms': [0.0833,0.25,0.4167], 'ems': [0.15,0.25,0.35],
                     'm': [0.25,0.5,0.75], 'lm': [0.1,0.5,0.9], 'mlm': [0.1667,0.5,0.8333],
                     'vm': [0.3333,0.5,0.6667], 'em': [0.4,0.5,0.6],
                     'ml': [0.5,0.75,1], 'lml': [0.35,0.75,1.15], 'mlml': [0.417,0.75,1.0833],
                     'vml': [0.5833,0.75,0.9167], 'eml': [0.65,0.75,0.85],
                     'l': [0.75,1,1.01], 'll': [0.6,1,1.01], 'mll': [0.667,1,1.01],
                     'vl': [0.833,1,1.01], 'el': [0.9,1,1.01]
        }
        '''
        self.shapes={'s': [-0.01,0,1.01], 
                     'l': [-0.01,1,1.01]
                     }
        self.shapes={'s': [-0.01,0,0.5], 
                     'm': [-0.01,0.5,1.01],
                     'l': [0.5,1,1.01]
                     }
        self.shapes={'s': [-0.01,0,0.25], 
                     'ms': [0,0.25,0.5],
                     'm': [0.25,0.5,0.75],
                     'ml': [0.5,0.75,1],
                     'l': [0.75,1,1.01]
                    }             
        self.shapes={'s': [-0.01,0,0.25], 'mls': [-0.01,0,0.33333],'vs': [-0.01,0,0.1667],
                     'ms': [0,0.25,0.5], 'mlms': [-0.0833,0.25,0.5833],'vms': [0.0833,0.25,0.4167], 
                     'm': [0.25,0.5,0.75], 'mlm': [0.1667,0.5,0.8333],'vm': [0.3333,0.5,0.6667], 
                     'ml': [0.5,0.75,1], 'mlml': [0.417,0.75,1.0833],'vml': [0.5833,0.75,0.9167], 
                     'l': [0.75,1,1.01], 'mll': [0.667,1,1.01],'vl': [0.833,1,1.01]
                    }
        
        '''
       
        '''
        self.shapes={'s': [-0.01,0,0.25], 'mls': [-0.01,0,0.33333],
                     'vs': [-0.01,0,0.1667], 'es': [-0.01,0,0.1],
                     'ms': [0,0.25,0.5],  'mlms': [-0.0833,0.25,0.5833],
                     'vms': [0.0833,0.25,0.4167],'ems': [0.15,0.25,0.35],
                         'ms_l': [0,0.25,10],  'mlms_l': [-0.0833,0.25,10],
                         'vms_l': [0.0833,0.25,10],'ems_l': [0.15,0.25,10],
                     'm': [0.25,0.5,0.75],  'mlm': [0.1667,0.5,0.8333],
                     'vm': [0.3333,0.5,0.6667],
                         'm_l': [0.25,0.5,10],  'mlm_l': [0.1667,0.5,10],
                         'vm_l': [0.3333,0.5,10], 
                     'ml': [0.5,0.75,1], 
                     'l': [0.75,1,1.01]
                    
        }
        '''
   
        print('load ok')
    
    #get MembershipFuction value of input value
    def getMF(self,value,shape): 
        MFValue=0
        if value<0 or value>1:
            return MFValue
        if shape=='dn':
           return 1 
        a=self.shapes[shape][0]
        b=self.shapes[shape][1]
        c=self.shapes[shape][2]
        if (value>=a and value<=b):
            MFValue=(value-a)/(b-a)
        if (value>=b and value<=c):
            MFValue=(c-value)/(c-b)
        return MFValue
           
    #get MembershipFuction value of input rule
    def getRuleMF(self,example,rule): 

        if len(example)!=len(rule):
            print("lengths are not same")
            return
        MFValue=1
        
        z=zip(example,rule)
        for tmpz in z:
            MFValue=min(MFValue,self.getMF(tmpz[0],tmpz[1]))    

        return MFValue
    
    def createRule(self,ruleLength,dnpro):
        rule=[]
        for i in range(ruleLength):
            rule.append(random.sample(self.shapes.keys(), 1)[0])
        for i in range(ruleLength):
            if random.random()<=dnpro:
                rule[i]='dn'           
        return rule
    
    def getSet(self):
        fset=random.sample(self.shapes.keys(), 1)      
        return fset[0]
    
    #get the number of conditons in a input rule
    def getRuleConditons(self,rule): 

        count=0
        
        for tmp in rule:
            if tmp!='dn':
                count=count+1    
        
        return count
    