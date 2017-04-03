import matplotlib.pyplot as plt

import cv2

import numpy as np

from sklearn import metrics
import itertools

def plotConfusionMatrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues, plot=True):
    """
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    spacing = ((len(str(cm).split("\n")[-1])-2) / cm.shape[1]) - 1
    print(" "+ (" "*spacing) + (" "*spacing).join(str(c) for c in classes))
    print(cm)

    if plot:
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

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

def leaveOnePersonOutSplits(people):
    splits = []
    for i in np.unique(people):
        splits.append((people != i, people == i))
    return splits

def resizeData(data, w):
    """
    data: array of shape (n,origw,origw,d)
    """
    n,origw,_,d = data.shape
    to_resize = data.transpose((1,2,0,3)).reshape((origw,origw,-1))
    resized = np.empty((w,w,n*d))
    for i in range(n*d):
        resized[:,:,i] = cv2.resize(to_resize[:,:,i],(w,w),interpolation=cv2.INTER_AREA)
    return resized.reshape((w,w,n,d)).transpose((2,0,1,3))

def resizeImages(originalSideLen, resizeFactor, X):
    X = X.reshape((X.shape[0], originalSideLen, originalSideLen, -1))
    Xresized = resizeData(X, int(originalSideLen*resizeFactor))
    return Xresized.reshape((X.shape[0],-1))

def getFeaturesFromSpecs(primaryDataFile, featureSpecs):
    primaryData = np.load(primaryDataFile)
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
            curData = resizeImages(128, scale, curData)
        print "Source ", source, " processed: shape ", curData.shape
        allFeatures.append(curData)
    X = np.concatenate(allFeatures, 1)
    return X

