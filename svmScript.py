import argparse

import itertools

import matplotlib.pyplot as plt

import numpy as np

import util

from sklearn import metrics
from sklearn.svm import SVC
from sklearn.utils import shuffle

def selectParamLinear(XTrain, yTrain, peopleTrain):
    CRange = 10.0**np.arange(-4,0)
    bestC = 0
    bestPerformance = 0
    for c in CRange:
        clf = SVC(kernel='linear', C=c)
        performance = cvPerformance(clf, XTrain, yTrain, peopleTrain)
        print "C = " + str(c) + ", accuracy = " + str(performance)
        if performance > bestPerformance:
            bestPerformance = performance
            bestC = c

    return bestC

def selectParamRBF(XTrain, yTrain, peopleTrain):
    bestPerformance = 0
    bestTuple = (0, 0)
    gammaRange = 10.0**np.arange(-4, -1)
    CRange = 10.0**np.arange(0, 4)

    for gamma in gammaRange:
        for c in CRange:
            clf = SVC(kernel='rbf', C=c, gamma=gamma)
            score = cvPerformance(clf, XTrain, yTrain, peopleTrain)
            print "gamma = " + str(gamma) + ", C = " + str(c) + ", accuracy = " + str(score)
            if score > bestPerformance:
                bestPerformance = score
                bestTuple = (gamma, c)

    return bestTuple


def main(dataFile):
    color, depth, labels, people = util.loadData(dataFile)

    n, colorFeatures = color.shape
    _, depthFeatures = depth.shape

    color, depth, labels, people = shuffle(color, depth, labels, people, random_state=0)

    XColorAndDepth = np.concatenate((color, depth), axis=1)
    XDepth = depth
    XColor = color
    y = labels

    XTrainColor, XTestColor, yTrainColor, yTestColor, peopleTrainColor, peopleTestColor = util.leaveOnePersonOut(3, XColor, y, people)

    XTrainDepth, XTestDepth, yTrainDepth, yTestDepth, peopleTrainDepth, peopleTestDepth = util.leaveOnePersonOut(3, XDepth, y, people)

    XTrainColorAndDepth, XTestColorAndDepth, yTrainColorAndDepth, yTestColorAndDepth, peopleTrainColorAndDepth, peopleTestColorAndDepth = util.leaveOnePersonOut(3, XColorAndDepth, y, people)

    # print "Selecting linear parameters for just color"
    # c = selectParamLinear(XTrainColor, yTrainColor, peopleTrainColor)
    # clf = SVC(kernel='linear', C=c)
    # clf.fit(XTrainColor, yTrainColor)
    # yPred = clf.predict(XTestColor)
    # score = metrics.accuracy_score(yTestColor, yPred)
    # print "Selected C = " + str(c) + ", accuracy = " + str(score)

    # print "Selecting linear parameters for just depth"
    # c = selectParamLinear(XTrainDepth, yTrainDepth, peopleTrainDepth)
    # clf = SVC(kernel='linear', C=c)
    # clf.fit(XTrainDepth, yTrainDepth)
    # yPred = clf.predict(XTestDepth)
    # score = metrics.accuracy_score(yTestDepth, yPred)
    # print "Selected C = " + str(c) + ", accuracy = " + str(score)

    # print "Selecting linear parameters for color and depth"
    # c = selectParamLinear(XTrainColorAndDepth, yTrainColorAndDepth, peopleTrainColorAndDepth)
    # clf = SVC(kernel='linear', C=c)
    # clf.fit(XTrainColorAndDepth, yTrainColorAndDepth)
    # yPred = clf.predict(XTestColorAndDepth)
    # score = metrics.accuracy_score(yTestColorAndDepth, yPred)
    # print "Selected C = " + str(c) + ", accuracy = " + str(score)

    # print "Selecting rbf parameters for just color"
    # gamma, c = selectParamRBF(XTrainColor, yTrainColor, peopleTrainColor)
    # clf = SVC(kernel='rbf', C=c, gamma=gamma)
    # clf.fit(XTrainColor, yTrainColor)
    # yPred = clf.predict(XTestColor)
    # score = metrics.accuracy_score(yTestColor, yPred)
    # print "Selected C = " + str(c) + ", gamma = " + str(gamma) + ", accuracy = " + str(score)

    # print "Selecting rbf parameters for just depth"
    # gamma, c = selectParamRBF(XTrainDepth, yTrainDepth, peopleTrainDepth)
    # clf = SVC(kernel='rbf', C=c, gamma=gamma)
    # clf.fit(XTrainDepth, yTrainDepth)
    # yPred = clf.predict(XTestDepth)
    # score = metrics.accuracy_score(yTestDepth, yPred)
    # print "Selected C = " + str(c) + ", gamma = " + str(gamma) + ", accuracy = " + str(score)

    # print "Selecting rbf parameters for color and depth"
    # gamma, c = selectParamRBF(XTrainColorAndDepth, yTrainColorAndDepth, peopleTrainColorAndDepth)
    # clf = SVC(kernel='rbf', C=c, gamma=gamma)
    # clf.fit(XTrainColorAndDepth, yTrainColorAndDepth)
    # yPred = clf.predict(XTestColorAndDepth)
    # score = metrics.accuracy_score(yTestColorAndDepth, yPred)
    # print "Selected C = " + str(c) + ", gamma = " + str(gamma) + ", accuracy = " + str(score)

    print "Fitting linear SVC with C=1"
    linearClf = SVC(kernel='rbf', C=100, gamma=0.0001)
    linearClf.fit(XTrainDepth, yTrainDepth)
    yPred = linearClf.predict(XTestDepth)
    score = metrics.accuracy_score(yTestDepth, yPred)
    cm = metrics.confusion_matrix(yTestDepth, yPred)
    plt.figure()
    print cm
    util.plotConfusionMatrix(cm, classes=np.unique(yTestDepth).tolist(),title="Confusion Matrix")
    print "Score: " + str(score)
    plt.show()

    # shrink depth
    # depth + nnnet
    # random forests



parser = argparse.ArgumentParser(description='Run SVM')
parser.add_argument('dataFile', help='Data file')

if __name__ == '__main__':
    namespace = parser.parse_args()
    args = vars(namespace)
    main(**args)