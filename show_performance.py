import numpy as np
import util
import argparse
from sklearn.externals import joblib
from sklearn import metrics
import matplotlib.pyplot as plt

def topKAccuracy(yProbabilities, yTrue, k, classes):
    correct = 0
    for probs, label in zip(yProbabilities, yTrue):
        for i in range(k):
            maxIndex = probs.argmax(axis=0)
            probs[maxIndex] = 0
            predictedLabel = classes[maxIndex]
            if predictedLabel == label:
                correct += 1
                break

    return correct/float(len(yTrue))



def main(dataFile, clfFile, plot_cm, leave_out, k):
    primaryData = np.load(dataFile)
    labels = primaryData['label']
    people = primaryData['person']

    clfdata = joblib.load(clfFile)
    clf = clfdata['clf']
    spec = clfdata['featureSpecs']

    data = util.getFeaturesFromSpecs(dataFile, spec)
    XTrain, XTest, yTrain, yTest, _, _ = util.leaveOnePersonOut(leave_out, data, labels, people)

    yPred = clf.predict(XTrain)
    score = metrics.accuracy_score(yTrain, yPred)
    print "Training Accuracy:", score
    print "Training set size:", yTrain.shape
    cm = metrics.confusion_matrix(yTrain, yPred)

    if plot_cm:
        plt.figure()
    util.plotConfusionMatrix(cm, classes=[chr(ord('a')+c) for c in np.unique(yTrain)],title="Confusion Matrix", plot=plot_cm)
    if plot_cm:
        plt.show()

    yPred = clf.predict(XTest)
    score = metrics.accuracy_score(yTest, yPred)
    topKScore = topKAccuracy(clf.predict_proba(XTest), yTest, k, clf.classes_)
    print "Test Accuracy:", score
    print "Top k accuracy for k = " + str(k) + ": " + str(topKScore)
    print "Testing set size:", yTest.shape
    cm = metrics.confusion_matrix(yTest, yPred)

    if plot_cm:
        plt.figure()
    util.plotConfusionMatrix(cm, classes=[chr(ord('a')+c) for c in np.unique(yTest)],title="Confusion Matrix", plot=plot_cm)
    if plot_cm:
        plt.show()

parser = argparse.ArgumentParser(description='Run SVM')
parser.add_argument('dataFile', help='Main data file')
parser.add_argument('clfFile', help='Classifier file')
parser.add_argument('--leave-out', type=int, default=3, help='Index of person to leave out')
parser.add_argument('--disable-cm', dest='plot_cm', action='store_false', help='Do not plot a confusion matrix')
parser.add_argument('--k', type=int, default=2, help='Value of k for top-k accuracy reporting')

if __name__ == '__main__':
    namespace = parser.parse_args()
    args = vars(namespace)
    main(**args)