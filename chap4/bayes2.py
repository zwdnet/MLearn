# -*- coding:utf-8 -*-
import random

import numpy as np
import re


def loadDataSet():
    postingList = postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]
    return postingList, classVec


def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print("The word: %s is not in my Vocabulary!" % word)
    return returnVec


def createVocabList(dataSet):
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document)
    return list(vocabSet)


def trainNB0(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory) / float(numTrainDocs)
    p0Num = np.ones(numWords);
    p1Num = np.ones(numWords)
    p0Denom = 2.0;
    p1Denom = 2.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = np.log(p1Num / p1Denom)
    p0Vect = np.log(p0Num / p0Denom)
    return p0Vect, p1Vect, pAbusive


def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec) + np.log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + np.log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0


def testingNB():
    list0Posts, listClasses = loadDataSet()
    myVocabList = createVocabList(list0Posts)
    trainMat = []
    for postinDoc in list0Posts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    p0V, p1V, pAb = trainNB0(np.array(trainMat), np.array(listClasses))
    testEntry = ["love", "my", "dalmation"]
    thisDoc = np.array(setOfWords2Vec(myVocabList, testEntry))
    if classifyNB(thisDoc, p0V, p1V, pAb):
        print(testEntry, "属于侮辱类")
    else:
        print(testEntry, "属于非侮辱类")
    testEntry = ["stupid", "garbage"]
    thisDoc = np.array(setOfWords2Vec(myVocabList, testEntry))
    if classifyNB(thisDoc, p0V, p1V, pAb):
        print(testEntry, "属于侮辱类")
    else:
        print(testEntry, "属于非侮辱类")


def textParse(bigString):
    listOfTokens = re.split(r'\W*', bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]


def bagOfWords2VecMN(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec


def spamTest():
    docList = []; classList = []; fullText = []
    for i in range(1, 26):
        wordList = textParse(open('email/spam/%d.txt' % i, 'r').read())
        docList.append(wordList)
        fullText.append(wordList)
        classList.append(1)
        wordList = textParse(open('email/ham/%d.txt' % i, 'r').read())
        docList.append(wordList)
        fullText.append(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)
    trainingSet = list(range(50))
    testSet = []
    for i in range(10):
        randIndex = int(random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat = []; trainClasses = []
    for docIndex in trainingSet:
        trainMat.append(setOfWords2Vec(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V, p1V, pSpam = trainNB0(np.array(trainMat), np.array(trainClasses))
    errorCount = 0
    for docIndex in testSet:
        wordVector = setOfWords2Vec(vocabList, docList[docIndex])
        if classifyNB(np.array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1
            print("分类错误的测试集:", docList[docIndex])
    print("错误率:%.2f%%" % (float(errorCount)/len(testSet)*100))

if __name__ == "__main__":
    # postingList, classVec = loadDataSet()
    # # for each in postingList:
    # #     print(each)
    # # print(classVec)
    # print("postingList:\n", postingList)
    # myVocabList = createVocabList(postingList)
    # print("myVocabList:\n", myVocabList)
    # trainMat = []
    # for postinDoc in postingList:
    #     trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    # print("TrainMat:\n", trainMat)
    # p0V, p1V, pAb = trainNB0(trainMat, classVec)
    # print("p0V:\n", p0V)
    # print("p1V:\n", p1V)
    # print("classVec:\n", classVec)
    # print("pAb:\n", pAb)
    # testingNB()
    # docList = [];
    # classList = []
    # for i in range(1, 26):
    #     wordList = textParse(open('email/spam/%d.txt' % i, 'r').read())
    #     docList.append(wordList)
    #     classList.append(1)
    #     wordList = textParse(open('email/ham/%d.txt' % i, 'r').read())
    #     print(wordList)
    #     docList.append(wordList)
    #     classList.append(0)
    # vocabList = createVocabList(docList)
    # print(vocabList)
    spamTest()