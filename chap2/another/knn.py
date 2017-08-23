# -*- coding:utf-8 -*-
import numpy as np
import operator
from matplotlib.font_manager import FontProperties
import matplotlib.lines as mlines
import matplotlib.pyplot as plt


def createDataSet():
    group = np.array([[1, 101], [5, 89], [108, 5], [115, 8]])
    labels = ['爱情片', '爱情片', '动作片', '动作片']
    return group, labels


def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    sortedDistIndices = distances.argsort()
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndices[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def file2matrix(filename):
    fr = open(filename)
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)
    returnMat = np.zeros((numberOfLines, 3))
    classLabelVector = []
    index = 0
    for line in arrayOLines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[0:3]
        if listFromLine[-1] == 'didntLike':
            classLabelVector.append(1)
        elif listFromLine[-1] == 'smallDoses':
            classLabelVector.append(2)
        elif listFromLine[-1] == 'largeDoses':
            classLabelVector.append(3)
        index += 1
    return returnMat, classLabelVector


def showdatas(datingDataMat, datingLabels):
    font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)
    fig, axs = plt.subplots(nrows=2, ncols=2, sharex=False, sharey=False, figsize=(13,8))

    numberOfLabels = len(datingLabels)
    LabelsColors = []
    for i in datingLabels:
        if i == 1:
            LabelsColors.append('black')
        if i == 2:
            LabelsColors.append('orange')
        if i == 3:
            LabelsColors.append('red')
    axs[0][0].scatter(x=datingDataMat[:,0], y=datingDataMat[:,1], color=LabelsColors, s=15, alpha=0.5)
    axs0_title_text = axs[0][0].set_title('每年获得的飞行常客里程数与玩视频游戏所消耗时间占比', FontProperties=font)
    axs0_xlabel_text = axs[0][0].set_xlabel('每年所获得的飞行常客里程数', FontProperties=font)
    axs0_ylabel_text = axs[0][0].set_ylabel('玩视频游戏所消耗时间', FontProperties=font)
    plt.setp(axs0_title_text, size = 9, weight = 'bold', color = 'red')
    plt.setp(axs0_xlabel_text, size = 7, weight = 'bold', color = 'black')
    plt.setp(axs0_ylabel_text, size = 7, weight = 'bold', color = 'black')

    axs[0][1].scatter(x=datingDataMat[:,0], y=datingDataMat[:,2], color=LabelsColors, s=15, alpha=0.5)
    axs1_title_text = axs[0][1].set_title('每年获得的飞行常客里程数与冰淇淋消耗量', FontProperties=font)
    axs1_xlabel_text = axs[0][1].set_xlabel('每年所获得的飞行常客里程数', FontProperties=font)
    axs1_ylabel_text = axs[0][1].set_ylabel('冰淇淋消耗量', FontProperties=font)
    plt.setp(axs1_title_text, size = 9, weight = 'bold', color = 'red')
    plt.setp(axs1_xlabel_text, size = 7, weight = 'bold', color = 'black')
    plt.setp(axs1_ylabel_text, size = 7, weight = 'bold', color = 'black')

    axs[1][0].scatter(x=datingDataMat[:,1], y=datingDataMat[:,2], color=LabelsColors, s=15, alpha=0.5)
    axs2_title_text = axs[1][0].set_title('玩视频游戏所消耗时间占比与冰淇淋消耗量', FontProperties=font)
    axs2_xlabel_text = axs[1][0].set_xlabel('视频游戏消耗时间', FontProperties=font)
    axs2_ylabel_text = axs[1][0].set_ylabel('冰淇淋消耗量', FontProperties=font)
    plt.setp(axs2_title_text, size = 9, weight = 'bold', color = 'red')
    plt.setp(axs2_xlabel_text, size = 7, weight = 'bold', color = 'black')
    plt.setp(axs2_ylabel_text, size = 7, weight = 'bold', color = 'black')

    didntLike = mlines.Line2D([], [], color = 'black', marker = '.',
                              markersize = 6, label = 'didntLike')
    smallDoses = mlines.Line2D([], [], color = 'orange', marker = '.',
                               markersize = 6, label = 'smallDoses')
    largeDoses = mlines.Line2D([], [], color = 'red', marker = '.',
                               markersize = 6, label = 'largeDoese')

    axs[0][0].legend(handles = [didntLike, smallDoses, largeDoses])
    axs[0][1].legend(handles = [didntLike, smallDoses, largeDoses])
    axs[1][0].legend(handles = [didntLike, smallDoses, largeDoses])

    plt.show()

def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)

    ranges = maxVals - minVals
    normDataSet = np.zeros(np.shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - np.tile(minVals, (m, 1))
    normDataSet = normDataSet / np.tile(ranges, (m, 1))
    return normDataSet, ranges, minVals

def datingClassTest():
    filename = "datingTestSet.txt"
    datingDataMat, datingLabels = file2matrix(filename)
    hoRatio = 0.5
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCount = 0.0

    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :],
                                     datingLabels[numTestVecs:m], 4)
        print("分类结果:%d\t真实类别:%d" % (classifierResult, datingLabels[i]))
        if classifierResult != datingLabels[i]:
            errorCount += 1.0
    print("错误率:%f%%" % (errorCount/float(numTestVecs)*100))


def classifyPerson():
    resultList = ['讨厌', '有些喜欢', '非常喜欢']
    precentTats = float(input("玩视频游戏时间占比"))
    ffMiles = float(input("每年飞行常旅客里程数"))
    iceCream = float(input("冰淇淋消耗"))

    filename = "datingTestSet.txt"
    datingDataMat, datingLabels = file2matrix(filename)
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = np.array([precentTats, ffMiles, iceCream])
    norminArr = (inArr - minVals) / ranges
    classifierResult = classify0(norminArr, normMat, datingLabels, 3)
    print(classifierResult)
    print("你可能%s这个人" % (resultList[classifierResult%3]))


from os import listdir
from sklearn.neighbors import KNeighborsClassifier as kNN


def img2vector(filename):
    returnVect = np.zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32*i+j] = int(lineStr[j])
    return returnVect

def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir('trainingDigits')
    m = len(trainingFileList)
    trainingMat = np.zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        classNumber = int(fileNameStr.split('_')[0])
        hwLabels.append(classNumber)
        trainingMat[i,:] = img2vector('trainingDigits/%s' % (fileNameStr))
    neigh = kNN(n_neighbors = 3, algorithm = 'auto')
    neigh.fit(trainingMat, hwLabels)
    testFileList = listdir('testDigits')
    errorCount1 = 0.0
    errorCount2 = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        classNumber = int(fileNameStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s' % (fileNameStr))
        classifierResult1 = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        classifierResult2 = neigh.predict(vectorUnderTest)
        print("分类返回的结果为%d\t真实结果为%d" % (classifierResult2, classNumber))
        if (classifierResult1 != classNumber):
            errorCount1 += 1.0
        if (classifierResult2 != classNumber):
            errorCount2 += 1.0
    print("自己的算法：总共错了%d个数据，错误率%f%%" % (errorCount1, errorCount1/mTest*100.0))
    print("skilearn算法：总共错了%d个数据，错误率%f%%" % (errorCount2, errorCount2/mTest*100.0))




if __name__ == '__main__':
    # group, labels = createDataSet()
    # test = [101, 20]
    # test_class = classify0(test, group, labels, 3)
    # print(test_class)
    # filename = 'datingTestSet.txt'
    # datingDataMat, datingLabels = file2matrix(filename)
    # print(datingDataMat, datingLabels)
    # showdatas(datingDataMat, datingLabels)
    # normDataSet, ranges, minVals = autoNorm(datingDataMat)
    # print(normDataSet)
    # print(ranges)
    # print(minVals)
    # datingClassTest()
    # classifyPerson()
    handwritingClassTest()
