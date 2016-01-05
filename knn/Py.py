from numpy import *
import matplotlib
import matplotlib.pyplot as plt
from knn import simpleKNN
from knn import handwriting
from knn import dating



group,labels= simpleKNN.createDateset()

testX = array([0.1, 0.3])


# print("测试结果是：" + simpleKNN.classify0(testX, group, labels, 3))

# print(handwriting.testHandWritingClass())


datingDataMat,datingLabels = dating.file2matrix()

print(datingDataMat)

fig = plt.figure()

ax = fig.add_subplot(111)

ax.scatter(datingDataMat[:, 1], datingDataMat[:, 2],15.0*array(datingLabels),15.0*array(datingLabels))

# plt.show()

a,b,c = dating.autoNorm(datingDataMat)

print(a)
print(b)
print(c)
print(datingLabels)

dating.datingClassTest()