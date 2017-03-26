import argparse

import itertools

import matplotlib.pyplot as plt

import numpy as np

from sklearn import metrics
from sklearn.svm import SVC
from sklearn.utils import shuffle

def plotConfusionMatrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def loadData(dataFile):
    loadedFile = np.load(dataFile)
    color = loadedFile['color']
    depth = loadedFile['depth']
    labels = loadedFile['label']
    people = loadedFile['person']
    return color, depth, labels, people

def leaveOnePersonOut(personToLeaveOut, X, y, people):
    n, _ = X.shape
    XTrain = []
    XTest = []
    yTrain = []
    yTest = []
    peopleTrain = []
    peopleTest = []

    for i in range(n):
        example = X[i, :]
        label = y[i]
        person = people[i]

        if person == personToLeaveOut:
            XTest.append(example)
            yTest.append(label)
            peopleTest.append(person)
        else:
            XTrain.append(example)
            yTrain.append(label)
            peopleTrain.append(person)

    return np.array(XTrain), np.array(XTest), np.array(yTrain), np.array(yTest), np.array(peopleTrain), np.array(peopleTest)

def cvPerformance(clf, XTrain, yTrain, peopleTrain):
    scores = []

    for i in np.unique(peopleTrain):
        XTrainCV, XTestCV, yTrainCV, yTestCV, _, _ = leaveOnePersonOut(i, XTrain, yTrain, peopleTrain)
        clf.fit(XTrainCV, yTrainCV)
        yPred = clf.predict(XTestCV)
        performance = metrics.accuracy_score(yTestCV, yPred)
        scores.append(performance)

    return np.array(scores).mean()

def selectParamLinear(XTrain, yTrain, peopleTrain):
    CRange = 10.0**np.arange(-3,3)
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
    gammaRange = 10.0**np.arange(-3, 3)
    CRange = 10.0**np.arange(-3, 3)

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
    color, depth, labels, people = loadData(dataFile)

    n, colorFeatures = color.shape
    _, depthFeatures = depth.shape

    color, depth, labels, people = shuffle(color, depth, labels, people, random_state=0)

    XColorAndDepth = np.concatenate((color, depth), axis=1)
    XDepth = depth
    XColor = color
    y = labels

    XTrainColor, XTestColor, yTrainColor, yTestColor, peopleTrainColor, peopleTestColor = leaveOnePersonOut(3, XColor, y, people)

    XTrainDepth, XTestDepth, yTrainDepth, yTestDepth, peopleTrainDepth, peopleTestDepth = leaveOnePersonOut(3, XDepth, y, people)

    XTrainColorAndDepth, XTestColorAndDepth, yTrainColorAndDepth, yTestColorAndDepth, peopleTrainColorAndDepth, peopleTestColorAndDepth = leaveOnePersonOut(3, XColorAndDepth, y, people)

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

    print "Selecting rbf parameters for just color"
    gamma, c = selectParamRBF(XTrainColor, yTrainColor, peopleTrainColor)
    clf = SVC(kernel='rbf', C=c, gamma=gamma)
    clf.fit(XTrainColor, yTrainColor)
    yPred = clf.predict(XTestColor)
    score = metrics.accuracy_score(yTestColor, yPred)
    print "Selected C = " + str(c) + ", gamma = " + str(gamma) + ", accuracy = " + str(score)

    print "Selecting rbf parameters for just depth"
    gamma, c = selectParamRBF(XTrainDepth, yTrainDepth, peopleTrainDepth)
    clf = SVC(kernel='rbf', C=c, gamma=gamma)
    clf.fit(XTrainDepth, yTrainDepth)
    yPred = clf.predict(XTestDepth)
    score = metrics.accuracy_score(yTestDepth, yPred)
    print "Selected C = " + str(c) + ", gamma = " + str(gamma) + ", accuracy = " + str(score)

    print "Selecting rbf parameters for color and depth"
    gamma, c = selectParamRBF(XTrainDepthAndColor, yTrainDepthAndColor, peopleTrainDepthAndColor)
    clf = SVC(kernel='rbf', C=c, gamma=gamma)
    clf.fit(XTrainDepthAndColor, yTrainDepthAndColor)
    yPred = clf.predict(XTestDepthAndColor)
    score = metrics.accuracy_score(yTestDepthAndColor, yPred)
    print "Selected C = " + str(c) + ", gamma = " + str(gamma) + ", accuracy = " + str(score)


    # print "Fitting linear SVC with C=1"
    # linearClf = SVC(kernel='linear', C=1)
    # linearClf.fit(XTrainColorAndDepth, yTrainColorAndDepth)
    # yPred = linearClf.predict(XTestColorAndDepth)
    # score = metrics.accuracy_score(yTestColorAndDepth, yPred)
    # cm = metrics.confusion_matrix(yTestColorAndDepth, yPred)
    # plt.figure()
    # print cm
    # plotConfusionMatrix(cm, classes=np.unique(yTestColorAndDepth).tolist(),title="Confusion Matrix")
    # print "Score: " + str(score)
    # plt.show()



parser = argparse.ArgumentParser(description='Run SVM')
parser.add_argument('dataFile', help='Data file')

if __name__ == '__main__':
    namespace = parser.parse_args()
    args = vars(namespace)
    main(**args)