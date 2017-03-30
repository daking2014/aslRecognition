import argparse
import cv2
import itertools
import matplotlib.pyplot as plt
import numpy as np
import util

from sklearn import metrics
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier

import scipy.stats

def selectParamSVMLinear(XTrain, yTrain, peopleTrain):
    CRange = 10.0**np.arange(-4,0)
    bestC = 0
    bestPerformance = 0
    for c in CRange:
        clf = SVC(kernel='linear', C=c)
        performance = util.cvPerformance(clf, XTrain, yTrain, peopleTrain)
        print "C = " + str(c) + ", accuracy = " + str(performance)
        if performance > bestPerformance:
            bestPerformance = performance
            bestC = c

    return bestC

def selectParamSVMRBF(XTrain, yTrain, peopleTrain):
    bestPerformance = 0
    bestTuple = (0, 0)
    gammaRange = 10.0**np.arange(-4, -1)
    CRange = 10.0**np.arange(0, 4)

    for gamma in gammaRange:
        for c in CRange:
            clf = SVC(kernel='rbf', C=c, gamma=gamma)
            score = util.cvPerformance(clf, XTrain, yTrain, peopleTrain)
            print "gamma = " + str(gamma) + ", C = " + str(c) + ", accuracy = " + str(score)
            if score > bestPerformance:
                bestPerformance = score
                bestTuple = (gamma, c)

    return bestTuple

def selectParamRandomForest(XTrain, yTrain, peopleTrain):
    params = {
        "n_estimators": scipy.stats.randint(5,20),
        "max_features": scipy.stats.uniform(0,1),
        "max_depth": scipy.stats.randint(1,10)
    }
    rscv = RandomizedSearchCV(
                RandomForestClassifier(),
                params,
                cv=util.leaveOnePersonOutSplits(peopleTrain),
                n_iter=20,
                verbose=1)
    rscv.fit(XTrain, yTrain)
    return rscv.best_params_

def selectParamLogReg(XTrain, yTrain, peopleTrain):
    CRange = 10.0**np.arange(-4,1)
    bestC = 0
    bestPerformance = 0
    for c in CRange:
        clf = LogisticRegression(C=c,multi_class="multinomial",solver="sag")
        performance = util.cvPerformance(clf, XTrain, yTrain, peopleTrain)
        print "C = " + str(c) + ", accuracy = " + str(performance)
        if performance > bestPerformance:
            bestPerformance = performance
            bestC = c

    return bestC

def selectParamKNN(XTrain, yTrain, peopleTrain):
    KRange = np.arange(3,11+1,2)
    bestK = 0
    bestPerformance = 0
    for k in KRange:
        clf = KNeighborsClassifier(k)
        performance = util.cvPerformance(clf, XTrain, yTrain, peopleTrain)
        print "k = " + str(k) + ", accuracy = " + str(performance)
        if performance > bestPerformance:
            bestPerformance = performance
            bestK = k

    return bestK

def main(dataFile, model, featureSpecs, plot_cm=True):
    """
    dataFile should be the path to a file with keys "label" and "person"
    dataSpecs should be a list of data specs, which are strings of one of these forms:
        "[key]" -> access this key of dataFile
        "pathtofile.npy" -> also access this path, which should be a numpy array
        "pathtofile.npz[key]" -> also access this key of this path, which should be a numpy zip
    optionally followed by "@scale" (which assumes 128x128) to start.
    i.e. to train on just color, should be [color]
         to train on just depth, [depth]
         to train on color and depth, [color] [depth]
         to train on shrunken depth, [depth]@0.2
         to train on overfeat depth, path_to_overfeat.npz[depth]
         to train on gabor, path_to_gabor.npy
    """
    primaryData = np.load(dataFile)
    labels = primaryData['label']
    people = primaryData['person']

    allFeatures = []
    for source in featureSpecs:
        name = source
        if "@" in name:
            name, scalestr = name.split("@")
            scale = float(scalestr)
        else:
            scale = 1.0
        if "[" in name:
            name, key = name[:-1].split("[")
        else:
            key = None
        if name is "":
            # Use primary data
            curData = primaryData
        else:
            # Load a file
            curData = np.load(name)
        if key is not None:
            curData = curData[key]
        if scale != 1.0:
            curData = util.resizeImages(128, scale, curData)
        print "Source ", source, " processed: shape ", curData.shape
        allFeatures.append(curData)
    X = np.concatenate(allFeatures, 1)
    n, d = X.shape
    print "Total shape:", X.shape

    X, y, people = shuffle(X, labels, people, random_state=0)

    XTrain, XTest, yTrain, yTest, peopleTrain, peopleTest = util.leaveOnePersonOut(3, X, y, people)

    if model == "SVMlinear":
        print "Selecting linear parameters"
        c = selectParamSVMLinear(XTrain, yTrain, peopleTrain)
        clf = SVC(kernel='linear', C=c)
        print "Selected C = " + str(c)
    elif model == "SVMrbf":
        gamma, c = selectParamSVMRBF(XTrain, yTrain, peopleTrain)
        clf = SVC(kernel='rbf', C=c, gamma=gamma)
        print "Selected C = " + str(c) + ", gamma = " + str(gamma)
    elif model == "RF":
        paramDict = selectParamRandomForest(XTrain, yTrain, peopleTrain)
        clf = RandomForestClassifier(**paramDict)
        print "Selected parameters ", paramDict
    elif model == "LogReg":
        c = selectParamLogReg(XTrain, yTrain, peopleTrain)
        clf = LogisticRegression(C=c,multi_class="multinomial",solver="sag")
        print "Selected C = " + str(c)
    elif model == "knn":
        k = selectParamKNN(XTrain, yTrain, peopleTrain)
        clf = KNeighborsClassifier(k)
    else:
        raise ValueError("Bad model " + model)
    
    clf.fit(XTrain, yTrain)

    yPred = clf.predict(XTest)
    score = metrics.accuracy_score(yTest, yPred)
    print "Accuracy:", score

    cm = metrics.confusion_matrix(yTest, yPred)
    print cm
    if plot_cm:
        plt.figure()
        util.plotConfusionMatrix(cm, classes=np.unique(yTest).tolist(),title="Confusion Matrix")
        plt.show()


parser = argparse.ArgumentParser(description='Run SVM')
parser.add_argument('dataFile', help='Main data file')
parser.add_argument('model', choices=['SVMlinear', 'SVMrbf', 'RF', 'LogReg', 'knn'], help='Model type')
parser.add_argument('featureSpecs', metavar="SPEC", nargs='+', help='Specifications for features to train on')
parser.add_argument('--disable-cm', dest='plot_cm', action='store_false', help='Do not plot a confusion matrix')

if __name__ == '__main__':
    namespace = parser.parse_args()
    args = vars(namespace)
    main(**args)
