from math import log
import operator
import pickle

# 计算给定数据集的香农熵
def calcShannonEntropy(dataSet):
    numEntries = len(dataSet)
    labelCounts={}
    # 为所有可能分类，创建字典
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel]=0
        labelCounts[currentLabel]+=1
    # 计算香农熵
    shannonEntropy = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        shannonEntropy -=prob * log(prob,2)
    return shannonEntropy



# 按照给定的特征划分数据集

def splitDataSet(dataSet, axis, value):
    retDataSet=[]
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet


# 选择最好的数据集划分方式
def chooseBestFeatureToSplit(dataSet):
    # 计算有多少特征属性
    numFeatures = len(dataSet[0])-1
    baseEntropy = calcShannonEntropy(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):
        # 计算第 i 个特征所有可能 取值的列表，然后转化为set集合
        featureList =[example[i] for example in dataSet]
        uniqueVals = set(featureList)
        newEntropy =0.0
        # 通过计算当前第 i个特征的每一个取值所构造的划分集合，得到第 i个特征对应的信息熵
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet,i,value)
            prob =len(subDataSet) / float(len(dataSet))
            newEntropy += prob * calcShannonEntropy(subDataSet)
        infoGain = baseEntropy - newEntropy
        if infoGain > bestInfoGain :
            bestInfoGain = infoGain
            bestFeature= i
    return bestFeature

# 创建决策树
def createTree(dataSet, labels):
    # 得到每一个实例所属的类别
    classList = [example[-1] for example in dataSet]
    # 类别完全相同，则停止划分
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    # 遍历完所有特征的时候返回出现次数最多
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    bestFeature = chooseBestFeatureToSplit(dataSet)
    bestFeatureLabel=  labels[bestFeature]
    myTree = {bestFeatureLabel:{}}
    del(labels[bestFeature])
    featValue =[example[bestFeature] for example in dataSet]
    uniqueVals = set(featValue)
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatureLabel][value] = createTree(splitDataSet(dataSet,bestFeature,value),subLabels)
    return myTree


def majorityCnt(classList):
    classCount={}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] =0
        classCount[vote]+=1
    sortedClassCount= sorted(classCount.iteritems(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]


# 分类
def classify(inputTree,featLabels, testVec):
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key],featLabels,testVec)
            else:
                classLabel = secondDict[key]
    return classLabel

# 存储模型
def storeTree(inputTree,filename):
    fw =open(filename,'wb')
    pickle.dump(inputTree,fw)
    fw.close()

# 加载模型
def grabTree(filename):
    fr =open(filename,'rb')
    return pickle.load(fr)

