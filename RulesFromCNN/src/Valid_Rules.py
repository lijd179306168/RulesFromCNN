# -*- coding: utf-8 -*-
import numpy as np
import random
from Triangle import *
from Data_CSV_Valid import *
from deap import base, creator, tools
from operator import attrgetter
ruleLength,dontCarePB=12,0.8
triangle = Triangle()
dataset = LoadDataCSVValid()
validatedata,validateeachClassAmout = dataset.get_validate_data()
classLabel=dataset.get_classLabel()
creator.create("FitnessMax", base.Fitness, weights=(1.0, ))
creator.create("MichiganRule", list, fitness=creator.FitnessMax,classLabel='-1',examples=[],correctCount=0,misCount=0)
toolbox = base.Toolbox()
toolbox.register("rules", triangle.createRule, ruleLength, dontCarePB)
toolbox.register("MichiganRule", tools.initIterate, creator.MichiganRule, toolbox.rules)
toolbox.register("population", tools.initRepeat, list, toolbox.MichiganRule)
# load_rules
with open('selectrule_12.csv','r')as f:
    f_csv = csv.reader(f)
    rules=[]
    for row in f_csv:
        rules.append(row)
popsize=len(rules)
pop1=toolbox.population(n=popsize)
for i in range(popsize):
    tmprule=rules[i][1:]
    for j in range(len(tmprule)):
        pop1[i][j]=tmprule[j]
    pop1[i].classLabel=rules[i][0:1][0]
def evaluate(rule):
    if rule.classLabel=='-1':
        fit=0
        return fit, 
    fit,correctcount,miscount=0,0,0
    rule.examples=[]
    for eachexample in validatedata:
        x=eachexample[0] 
        y=eachexample[1][0]
        id=eachexample[2][0]
        if triangle.getRuleMF(x,rule)>0.5:
            if y==rule.classLabel:
                correctcount=correctcount+1
                rule.examples.append(id)
            if y!=rule.classLabel:
                miscount=miscount+1 
    fit=correctcount/validateeachClassAmout[rule.classLabel]-0.2*miscount/validateeachClassAmout[rule.classLabel]
    rule.correctCount=correctcount
    rule.misCount=miscount
    return fit,

def outputelists(elistsize):
    conditionsCount=0                
    ruleCount=0             
    for i in range(elistsize):
        if pop1[i].fitness.values[0]>0:
            tmpCount=triangle.getRuleConditons(pop1[i])
            conditionsCount=conditionsCount+tmpCount
            ruleCount=ruleCount+1 
    print("conditionsCount",conditionsCount)
    print("ruleCount",ruleCount)

#begin
for eachrule in pop1:
    fitnessValue=evaluate(eachrule)
    eachrule.fitness.values = fitnessValue     
    
fvalid= open('valid-valid.csv','w',newline='')
fvalid_csv = csv.writer(fvalid)
correctcount,miscount,uncount=0,0,0
for eachexample in validatedata:
    x=eachexample[0]
    y=eachexample[1][0]  
    maxActivateValue=0
    maxMatchRule=None
    for eachrule in pop1:
        calcValue=triangle.getRuleMF(x,eachrule)
        if calcValue>maxActivateValue:
            maxActivateValue=calcValue
            maxMatchRule=eachrule
    if maxActivateValue<0.5:
        uncount=uncount+1      
    else:
        if y==maxMatchRule.classLabel:
            correctcount=correctcount+1
        if y!=maxMatchRule.classLabel:
            miscount=miscount+1  
print("len(validatedata)",len(validatedata))
print("correctcount/len(validatedata)",correctcount/len(validatedata))
print("miscount/len(validatedata)",miscount/len(validatedata))
print("uncount/len(validatedata)",uncount/len(validatedata))
row=[]
row.append(correctcount/len(validatedata))
row.append(miscount/len(validatedata))
row.append(uncount/len(validatedata))
fvalid_csv.writerow(row)
elistsize=len(pop1)
outputelists(elistsize)


#