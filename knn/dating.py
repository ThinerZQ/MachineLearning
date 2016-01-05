from numpy import *
import os
from knn import simpleKNN

def file2matrix():


    filename = "../resource/Dating/datingTestSet2.txt"
    fr = open(filename)
    numberOfLines = len(fr.readlines())         #get the number of lines in the file
    returnMat = zeros((numberOfLines,3))        #prepare matrix to return
    classLabelVector = []                       #prepare labels return
    fr = open(filename)
    index = 0
    for line in fr.readlines():
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat,classLabelVector


# 归一化处理
def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges =maxVals- minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m, 1))
    normDataSet = normDataSet/tile(ranges, (m ,1))
    return normDataSet,ranges,minVals


def datingClassTest():
    hoRatio =0.10
    datingDataMat ,datingLabels = file2matrix();
    normMat,ranges,minvals = autoNorm(datingDataMat)
    m=normMat.shape[0]
    numberTestVecs = int(m*hoRatio)
    errorCount =0
    for i in range(numberTestVecs):
        classifierResult = simpleKNN.classify0(normMat[i,:],normMat[numberTestVecs:m,:],datingLabels[numberTestVecs:m],5)

        if classifierResult != datingLabels[i] :
            print("the classifier came back with %d ,the real answer is :%d" %(classifierResult,datingLabels[i]))
            errorCount+=1.0

    print("the total error rate is :%f " %(errorCount/float(numberTestVecs)))