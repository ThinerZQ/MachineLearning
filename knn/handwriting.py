from numpy import *
import matplotlib
import matplotlib.pyplot as plt
import os


# 读取文件名对应的文件
def img2vector(filename):
    rows = 32
    cols = 32
    imgVector = zeros((1, rows * cols))
    fileIn = open(filename)
    for row in range(rows):
        lineStr = fileIn.readline()
        for col in range(cols):
            imgVector[0, row * 32 + col] = int(lineStr[col])

    # print(imgVector)
    return imgVector


# load dataSet
def loadDataSet():

    print("---Getting training set...")

    # 获取训练集和测试集的路径
    dataSetDir = os.getcwd()+'/resource/Handwriting Recognition/'

    # 得到训练集所在的目录下所有的文件列表
    trainingFileList = os.listdir(dataSetDir + 'trainingDigits')

    # 得到有多少个训练文件
    numSamples = len(trainingFileList)

    # 训练集
    train_x = zeros((numSamples, 1024))
    # 标签列表
    train_y = []

    # 开始读取训练集
    for i in range(numSamples):
        # 得到某一个文件名
        filename = trainingFileList[i]

        # 读取文件
        train_x[i, :] = img2vector(dataSetDir + 'trainingDigits/%s' % filename)

        # 得到这个文件对应的标签
        label = int(filename.split('_')[0])
        # 讲当前文件的标签 放入标签列表
        train_y.append(label)

    # step 2: Getting testing set
    print("---Getting testing set...")

    # 加载测试文件列表
    testingFileList = os.listdir(dataSetDir + 'testDigits')

    # 记录测试文件个数
    numSamples = len(testingFileList)
    # 生成测试文件矩阵
    test_x = zeros((numSamples, 1024))
    # 测试标签集合
    test_y = []

    for i in range(numSamples):
        filename = testingFileList[i]

        # 读取测试文件
        test_x[i, :] = img2vector(dataSetDir + 'testDigits/%s' % filename)

        # 得到测试文件的标签
        label = int(filename.split('_')[0]) # return 1
        test_y.append(label)

    # 返回训练集合，训练标签，测试集合，测试标签
    return train_x, train_y, test_x, test_y

# test hand writing class
def testHandWritingClass():

    # step 1: load data
    print ("step 1: load data...")
    train_x, train_y, test_x, test_y = loadDataSet()

    # step 2: training...
    print("step 2: training...")
    pass

    # step 3: testing
    print("step 3: testing...")
    # 得到测试文件的个数
    numTestSamples = test_x.shape[0]
    matchCount = 0
    for i in range(numTestSamples):
        predict = kNNClassify(test_x[i], train_x, train_y, 3)
        if predict == test_y[i]:
            matchCount += 1
    accuracy = float(matchCount) / numTestSamples

    # step 4: show the result
    print("step 4: show the result...")
    print('The classify accuracy is: %.2f%%' % (accuracy * 100))

# classify using kNN
def kNNClassify(newInput, dataSet, labels, k):

     # shape[0] 表示有多少行
    numSamples = dataSet.shape[0]

    # step 1: calculate Euclidean distance
    # tile(A, reps): Construct an array by repeating A reps times
    # the following copy numSamples rows for dataSet
    diff = tile(newInput, (numSamples, 1)) - dataSet # Subtract element-wise
    squaredDiff = diff ** 2 # squared for the subtract
    squaredDist = sum(squaredDiff, axis = 1) # sum is performed by row
    distance = squaredDist ** 0.5

    ## step 2: sort the distance
    # argsort() returns the indices that would sort an array in a ascending order
    sortedDistIndices = argsort(distance)

    classCount = {} # define a dictionary (can be append element)
    for i in range(k):
        ## step 3: choose the min k distance
        voteLabel = labels[sortedDistIndices[i]]

        ## step 4: count the times labels occur
        # when the key voteLabel is not in dictionary classCount, get()
        # will return 0
        classCount[voteLabel] = classCount.get(voteLabel, 0) + 1

    ## step 5: the max voted class will return
    maxCount = 0
    for key, value in classCount.items():
        if value > maxCount:
            maxCount = value
            maxIndex = key

    return maxIndex