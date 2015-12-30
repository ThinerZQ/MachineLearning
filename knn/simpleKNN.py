from numpy import *
import operator

def createDateset() :
    group = array([[1.0, 0.9], [1.0, 1.0], [0.2, 0.1], [0.2, 0.3]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize, 1))-dataSet
    # print(diffMat)
    # print(dataSet)


    sqDiffMat = diffMat**2
    # print(sqDiffMat)
    # 按照行求和
    distances = sqDiffMat.sum(1)
    # print(distances)

    # 对求出的和序列排序,返回的是数组值从小到大的索引值
    sortedDistIndicies = distances.argsort()
    # print(sortedDistIndicies)

    classCount={}
    # 生成 一个分类字典
    for i in range(k):
        voteILable = labels[sortedDistIndicies[i]]
        #print(voteILable)
        classCount[voteILable] =classCount.get(voteILable,0)+1

    # python内建排序函数， key 是一个匿名函数，operator.itemgetter(1)也可以实现匿名函数的功能，根据第classCount中第二个域元素排序
    sortedClassCount= sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    # print(sortedClassCount)
    return sortedClassCount[0][0]







