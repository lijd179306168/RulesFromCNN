# -*- coding: utf-8 -*-
import numpy as np
import random
from Triangle import *
from Data_CSV import *
from deap import base, creator, tools
from operator import attrgetter

ruleLength,dontCarePB=12,0.8
CXPB, MUTPB  = 0.5, 0.2
ElistPoolSize=10
ClassNumber=10

triangle = Triangle()
dataset = LoadDataCSV()
traindata,eachClassAmout = dataset.get_train_data()
validatedata,validateeachClassAmout = dataset.get_validate_data()
classLabel=dataset.get_classLabel()
creator.create("FitnessMax", base.Fitness, weights=(1.0, ))
creator.create("MichiganRule", list, fitness=creator.FitnessMax,classLabel='-1',examples=[],correctCount=0,misCount=0)
toolbox = base.Toolbox()
toolbox.register("rules", triangle.createRule, ruleLength, 1)
toolbox.register("MichiganRule", tools.initIterate, creator.MichiganRule, toolbox.rules)
toolbox.register("population", tools.initRepeat, list, toolbox.MichiganRule)
toolbox.register("mate", tools.cxTwoPoint) 
toolbox.register("select", tools.selTournament, tournsize=3) 

pop1=toolbox.population(n=300)

def assignClasstoRule():
    for eachrule in pop1:
        classCountDict=[]
        classCountDict=dict(zip(classLabel,[0 for i in range(len(classLabel))]))
        for eachexample in traindata:
            x=eachexample[0] 
            y=eachexample[1][0] 
            MFV=triangle.getRuleMF(x,eachrule)
            if MFV>0.5:
               classCountDict[y]=classCountDict[y]+1
        assignClass=max(classCountDict, key=classCountDict.get)
        eachrule.classLabel=assignClass 
    return 0
def evaluate(rule):
    if rule.classLabel=='-1':
        fit=0
        return fit, 
    fit,correctcount,miscount=0,0,0
    rule.examples=[]
    for eachexample in traindata:
        x=eachexample[0]
        y=eachexample[1][0]  
        id=eachexample[2][0]
        if triangle.getRuleMF(x,rule)>0.5:
            if y==rule.classLabel:
                correctcount=correctcount+1
                rule.examples.append(id)
            if y!=rule.classLabel:
                miscount=miscount+1 
    fit=correctcount/eachClassAmout[rule.classLabel]-0.5*miscount/eachClassAmout[rule.classLabel]  
    rule.correctCount=correctcount
    rule.misCount=miscount
    if miscount>20:
        fit=0
    return fit,
def notinClassLabel(classLabel1,classLabel2): 
    notinCount=0
    for letter in classLabel2:
        if letter not in  classLabel1:
            notinCount=notinCount+1
    return notinCount
def elistSelect(CountofEachRule=3):
    examplesbyClassDict=dict(zip(classLabel,[[] for i in range(len(classLabel))]))
    for eachrule in pop1:
        if eachrule.fitness.values[0]>0:
            y=eachrule.classLabel
            if y=='-1':
                continue
            examplesbyClassDict[y].append(eachrule)
    examplesbyClassDictSorted=dict(zip(classLabel,[[] for i in range(len(classLabel))]))
    for key, value in examplesbyClassDict.items():
        tmp=sorted(value,key=attrgetter("fitness"), reverse=True) 
        examplesbyClassDictSorted[key]=tmp     
    classindex=0
    for key in classLabel:
        if (len(examplesbyClassDictSorted[key])==0):
            classindex=classindex+1
            continue
        pop1[CountofEachRule*classindex]=examplesbyClassDictSorted[key][0]       
        existedElistCount=1
        while (existedElistCount<CountofEachRule):    
            maxnotin=0
            maxnotinrule=toolbox.MichiganRule() 
            maxnotinrule.fitness.values=(0,)     
            for i in range(1,len(examplesbyClassDictSorted[key])):
                eachrulenotin=100000   #max value 
                for j in range(0,existedElistCount):
                    tmpnotin=notinClassLabel(pop1[CountofEachRule*classindex+j].examples,examplesbyClassDictSorted[key][i].examples)
                    if tmpnotin<eachrulenotin:
                        eachrulenotin=tmpnotin
                        tmpnotinrule=examplesbyClassDictSorted[key][i]                  
                if eachrulenotin>maxnotin:
                    maxnotin=eachrulenotin
                    maxnotinrule=tmpnotinrule
            pop1[CountofEachRule*classindex+existedElistCount]=maxnotinrule
            existedElistCount=existedElistCount+1     
        classindex=classindex+1
def performance(g,ftrain_csv):
    correctcount,miscount,uncount=0,0,0
    for eachexample in traindata:
        x=eachexample[0] 
        y=eachexample[1][0]  
        maxActivateValue=0
        maxMatchRule=None    
        for eachrule in pop1:          
            if eachrule.fitness.values[0]==0:               
                continue
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
    print("correctcount/len(traindata)",correctcount/len(traindata))
    print("miscount/len(traindata)",miscount/len(traindata))
    print("uncount/len(traindata)",uncount/len(traindata))    
    row=[]
    row.append(g)
    row.append(correctcount/len(traindata))
    row.append(miscount/len(traindata))
    row.append(uncount/len(traindata))
    ftrain_csv.writerow(row)   
    return 0  
def outputelists(ElistPoolSize):
    for i in range(ElistPoolSize):
        print("rule classLabel %d="%i,pop1[i].classLabel)
        print("rule examples %d="%i,pop1[i].examples)
        print("rule %d="%i,pop1[i])
        print("rule fitness %d="%i,pop1[i].fitness)   
#begin
assignClasstoRule()
for eachrule in pop1:
    fitnessValue=evaluate(eachrule)
    eachrule.fitness.values = fitnessValue  
elistSelect(ElistPoolSize)
frule= open('rule.csv','w',newline='')
f_csv = csv.writer(frule)
ftrain= open('train.csv','w',newline='')
ftrain_csv = csv.writer(ftrain)
for g in range(200):
    if g % 10 == 0:
        print("-- Generation %i --" % g)
    offspring = toolbox.select(pop1, len(pop1))
    offspring = list(map(toolbox.clone, offspring))
    for child1, child2 in zip(offspring[::2], offspring[1::2]):
        if random.random() < CXPB:
            toolbox.mate(child1, child2)
            del child1.fitness.values
            del child2.fitness.values
    for mutant in offspring:
        for mpos in range(len(mutant)-1):
            if random.random() < MUTPB:
                if random.random() <dontCarePB:
                    mutant[mpos]='dn'
                else:
                    mutant[mpos]=triangle.getSet()
        del mutant.fitness.values
    assignClasstoRule()
    for eachrule in offspring:
        fitnessValue=evaluate(eachrule)
        eachrule.fitness.values = fitnessValue           
    elistSelect(ElistPoolSize)
    offspringbak = list(map(toolbox.clone, pop1))
    pop1[:] = offspring[:]   
    pop1[:ElistPoolSize*ClassNumber]=offspringbak[:ElistPoolSize*ClassNumber]  
    outputelists(ElistPoolSize*ClassNumber)  
    print("g",g)   
    if g % 2 == 0:
        performance(g,ftrain_csv) 
    row=[]
    row.append(g)
    f_csv.writerow(row)    
    for eachrule in pop1:
        row=[]
        if eachrule.fitness.values[0]>0:               
           row.append(eachrule.classLabel)
           for i in range(len(eachrule)):
               row.append(eachrule[i]) 
           f_csv.writerow(row)
