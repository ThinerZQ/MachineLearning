from numpy import *

from knn import simpleKNN
from knn import handwriting

group,labels= simpleKNN.createDateset()

testX = array([0.1, 0.3])


print("测试结果是：" + simpleKNN.classify0(testX, group, labels, 3))

print(handwriting.testHandWritingClass())
