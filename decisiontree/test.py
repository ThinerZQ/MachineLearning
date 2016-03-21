from decisiontree import Util
from decisiontree import DecisionTree
import copy
import os
from decisiontree import treePlotter
myData ,labels = Util.createDataSet()
# 深拷贝
newLabel =copy.deepcopy(labels)
print(labels)
tree = DecisionTree.createTree(myData,labels)
# print(myData)

# print(DecisionTree.calcShannonEntropy(myData))
# print(DecisionTree.splitDataSet(myData,0,1))
# print(DecisionTree.chooseBestFeatureToSplit(myData))

# print(tree)

print(myData)
print(newLabel)

classLabel = DecisionTree.classify(tree,newLabel,[1,1])
print(classLabel)

DecisionTree.storeTree(tree,'classifierStorage.txt')

treeTxt = DecisionTree.grabTree('classifierStorage.txt')

print(treeTxt)



fr = open('../resource/lenses/lenses.txt')

lenses =[inst.strip().split('\t') for inst in fr.readlines()]

lensesLables = ['age','prescript','astigmatic','tearRate']

lensesTree = DecisionTree.createTree(lenses,lensesLables)



print(lensesTree)