import matplotlib.pyplot as plt

import numpy as np

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