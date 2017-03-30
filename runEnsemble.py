import matplotlib.pyplot as plt
import numpy as np
import os
import util

from sklearn import metrics
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.externals import joblib

ENSEMBLE_FOLDER = 'classifiersForEnsemble'
DATA_FILE = 'bcmnw.npz'
GABOR_FILE = 'gabor_bcmnw.npy'
OVERFEAT_FILE = 'overfeat_bcmnw.npz'

def main():
    classifiers = []
    specs = []
    for filename in os.listdir(ENSEMBLE_FOLDER):
        clf = joblib.load(ENSEMBLE_FOLDER + "\\" + filename)
        classifiers.append(clf["clf"])
        specs.append(clf["featureSpecs"])

    primaryData = np.load(DATA_FILE)
    labels = primaryData['label']
    people = primaryData['person']

    sampleDatay = None
    specsToTestData = {}
    for spec in specs:
        if not specsToTestData.has_key(str(spec)):
            data = util.getFeaturesFromSpecs(DATA_FILE, spec)
            _, XTest, _, yTest, _, _ = util.leaveOnePersonOut(3, data, labels, people)
            specsToTestData[str(spec)] = [XTest, yTest]

            if sampleDatay == None:
                sampleDatay = yTest

    totalPredictions = np.zeros((len(sampleDatay), len(np.unique(sampleDatay))))
    for i in range(len(classifiers)):
        spec = str(specs[i])
        data = specsToTestData[spec]
        predictions = classifiers[i].predict_proba(data[0])
        totalPredictions = np.add(totalPredictions, predictions)

    # HARDCODED CLASS MAPPING
    classMapping = {0:1, 1:2, 2:12, 3:13, 4:22 }
    yPred = []
    for i in range(len(sampleDatay)):
        predictions = totalPredictions[i, :]
        classIndex = predictions.argmax(axis=0)
        yPred.append(classMapping[classIndex])

    score = metrics.accuracy_score(sampleDatay, yPred)
    print "Accuracy:", score

    cm = metrics.confusion_matrix(sampleDatay, yPred)
    print cm
    plt.figure()
    util.plotConfusionMatrix(cm, classes=np.unique(sampleDatay).tolist(),title="Confusion Matrix")
    plt.show()


if __name__ == '__main__':
    main()