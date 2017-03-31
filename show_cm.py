import numpy as np
import util
import argparse
from sklearn.externals import joblib
from sklearn import metrics
import matplotlib.pyplot as plt

def main(dataFile, clfFile):
    primaryData = np.load(dataFile)
    labels = primaryData['label']
    people = primaryData['person']

    clfdata = joblib.load(clfFile)
    clf = clfdata['clf']
    spec = clfdata['featureSpecs']

    data = util.getFeaturesFromSpecs(dataFile, spec)
    _, XTest, _, yTest, _, _ = util.leaveOnePersonOut(3, data, labels, people)

    yPred = clf.predict(XTest)
    cm = metrics.confusion_matrix(yTest, yPred)
    
    plt.figure()
    util.plotConfusionMatrix(cm, classes=[chr(ord('a')+c) for c in np.unique(yTest)],title="Confusion Matrix")
    plt.show()

parser = argparse.ArgumentParser(description='Run SVM')
parser.add_argument('dataFile', help='Main data file')
parser.add_argument('clfFile', help='Classifier file')

if __name__ == '__main__':
    namespace = parser.parse_args()
    args = vars(namespace)
    main(**args)