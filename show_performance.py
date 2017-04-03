import numpy as np
import util
import argparse
from sklearn.externals import joblib
from sklearn import metrics
import matplotlib.pyplot as plt

def main(dataFile, clfFile, plot_cm, leave_out):
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
    cm = metrics.confusion_matrix(yTrain, yPred)
    
    if plot_cm:
        plt.figure()
    util.plotConfusionMatrix(cm, classes=[chr(ord('a')+c) for c in np.unique(yTrain)],title="Confusion Matrix", plot=plot_cm)
    if plot_cm:
        plt.show()

    yPred = clf.predict(XTest)
    score = metrics.accuracy_score(yTest, yPred)
    print "Test Accuracy:", score
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

if __name__ == '__main__':
    namespace = parser.parse_args()
    args = vars(namespace)
    main(**args)