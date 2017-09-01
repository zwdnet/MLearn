# -*- coding:utf-8 -*-
from math import log
import operator

def createDataSet():
    dataSet = [[0, 0, 0, 0, 'no'],
               [0, 0, 0, 1, 'no'],
               [0, 1, 0, 1, 'yes'],
               [0, 1, 1, 0, 'yes'],
               [0, 0, 0, 0, 'no'],
               [1, 0, 0, 0, 'no'],
               [1, 0, 0, 1, 'no'],
               [1, 1, 1, 1, 'yes'],
               [1, 0, 1, 2, 'yes'],
               [1, 0, 1, 2, 'yes'],
               [2, 0, 1, 2, 'yes'],
               [2, 0, 1, 1, 'yes'],
               [2, 1, 0, 1, 'yes'],
               [2, 1, 0, 2, 'yes'],
               [2, 0, 0, 0, 'no']]
    labels = ['年龄', '有工作', '有自己的房子', '信誉情况']
    return dataSet, labels


def calcShannonEnt(dataSet):
    numEntires = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntires
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt

def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet


def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
        print("第%d个特征的增益为%.3f" % (i, infoGain))
        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature


def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
        sortedClassCount = sorted(classCount.items(), key = operator.itemgetter(1), reverse = True)
        return sortedClassCount[0][0]


def createTree(dataSet, labels, featLabels):
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    featLabels.append(bestFeatLabel)
    myTree = {bestFeatLabel:{}}
    del(labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), labels, featLabels)
    return myTree


def classify(inputTree, featLabels, testVec):
    firstStr = next(iter(inputTree))
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key], featLabels, testVec)
            else:
                classLabel = secondDict[key]
    return classLabel


import pickle

def storeTree(inputTree, filename):
    with open(filename, 'wb') as fw:
        pickle.dump(inputTree, fw)


def grabTree(filename):
    fr = open(filename, 'rb')
    return pickle.load(fr)


from sklearn import tree
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import pydotplus
from sklearn.externals.six import StringIO


if __name__ == '__main__':
    # dataSet, featrues = createDataSet()
    # print(dataSet)
    # print(calcShannonEnt(dataSet))
    # print("最优特征索引值:" + str(chooseBestFeatureToSplit(dataSet)))
    # featLabels = []
    # myTree = createTree(dataSet, featrues, featLabels)
    # print(myTree)
    # testVec = [0, 1]
    # result = classify(myTree, featLabels, testVec)
    # if result == "yes":
    #     print("放贷")
    # if result == "no":
    #     print("不放贷")
    # # storeTree(myTree, "classifierStorage.txt")
    # restoreTree = grabTree("classifierStorage.txt")
    # print(restoreTree)
    # fr = open('lenses.txt')
    # lenses = [inst.strip().split('\t') for inst in fr.readlines()]
    # print(lenses)
    with open('lenses.txt', 'r') as fr:
        lenses = [inst.strip().split('\t') for inst in fr.readlines()]
    lenses_target = []
    for each in lenses:
        lenses_target.append(each[-1])

    lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
    # clf = tree.DecisionTreeClassifier()
    # lenses = clf.fit(lenses, lensesLabels)
    lenses_list = []
    lenses_dict = {}
    for each_label in lensesLabels:
        for each in lenses:
            lenses_list.append(each[lensesLabels.index(each_label)])
        lenses_dict[each_label] = lenses_list
        lense_list = []
    print(lenses_dict)
    lenses_pd = pd.DataFrame(lenses_dict)
    print(lenses_pd)
    le = LabelEncoder()
    for col in lenses_pd.columns:
        lenses_pd[col] = le.fit_transform(lenses_pd[col])
    print(lenses_pd)

    clf = tree.DecisionTreeClassifier(max_depth=4)
    clf = clf.fit(lenses_pd.values.tolist(), lenses_target)
    dot_data = StringIo()
    tree.export_graphviz(clf, out_file=dot_data, feature_names=lenses_pd.keys(),
                         class_names=clf.classes_,
                         filled=True, rounded=True,
                         special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_pdf("tree.pdf")

