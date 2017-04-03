import matplotlib.pyplot as plt
import numpy as np
import os
import util
import argparse

from sklearn import metrics
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.externals import joblib



def probability_votes(clfs, specs, specMap, n_test, targets):
    totalPredictions = np.zeros((n_test, len(targets)))
    for i in range(len(clfs)):
        spec = str(specs[i])
        data = specMap[spec]
        predictions = clfs[i].predict_proba(data[0])
        totalPredictions = np.add(totalPredictions, predictions)

    yPred = []
    for i in range(n_test):
        predictions = totalPredictions[i, :]
        classIndex = predictions.argmax(axis=0)
        yPred.append(targets[classIndex])
    return yPred

def majority_votes(clfs, specs, specMap, n_test, targets):
    """
    preds: (num_clf,n_test) of best candidates
    Ties are broken by first classifier
    """
    num_clf = len(clfs)
    # Compute votes
    preds = np.empty((num_clf,n_test))
    for i,(clf,spec) in enumerate(zip(clfs,specs)):
        X_test, _ = specMap[str(spec)]
        preds[i,:] = clf.predict(X_test)
    # Find out which things got votes
    all_votes = np.empty((num_clf,n_test,len(targets)))
    for i,target in enumerate(targets):
        all_votes[:,:,i] = (preds == target)
    # Give preference to first classifier
    all_votes[0,:,:] *= 1.01
    # Add up the votes for each target
    nvotes = np.sum(all_votes,0) # of shape (n_test, len(targets))
    # Get the best target
    y_pred_idx = np.argmax(nvotes, 1) # of shape (n_test,)
    y_pred = targets[y_pred_idx]
    return y_pred

def main(ensemble_folder, data_file, leave_out, plot_cm, majority=False):
    classifiers = []
    specs = []
    for filename in os.listdir(ensemble_folder):
        clf = joblib.load(os.path.join(ensemble_folder, filename))
        classifiers.append(clf["clf"])
        specs.append(clf["featureSpecs"])
        print "Loaded classifier ", filename

    primaryData = np.load(data_file)
    labels = primaryData['label']
    people = primaryData['person']

    specsToTrainData = {}
    specsToTestData = {}
    for spec in specs:
        if not specsToTestData.has_key(str(spec)):
            data = util.getFeaturesFromSpecs(data_file, spec)
            XTrain, XTest, yTrain, yTest, _, _ = util.leaveOnePersonOut(leave_out, data, labels, people)
            specsToTestData[str(spec)] = [XTest, yTest]
            specsToTrainData[str(spec)] = [XTrain, yTrain]

    if majority:
        yPred = majority_votes(classifiers, specs, specsToTrainData, len(yTrain), np.unique(yTrain))
    else:
        yPred = probability_votes(classifiers, specs, specsToTrainData, len(yTrain), np.unique(yTrain))
    score = metrics.accuracy_score(yTrain, yPred)
    print "Train Accuracy:", score

    cm = metrics.confusion_matrix(yTrain, yPred)
    if plot_cm:
        plt.figure()
    util.plotConfusionMatrix(cm, classes=[chr(ord('a')+c) for c in np.unique(yTrain)],title="Confusion Matrix", plot=plot_cm)
    if plot_cm:
        plt.show()

    if majority:
        yPred = majority_votes(classifiers, specs, specsToTestData, len(yTest), np.unique(yTest))
    else:
        yPred = probability_votes(classifiers, specs, specsToTestData, len(yTest), np.unique(yTest))
    score = metrics.accuracy_score(yTest, yPred)
    print "Test Accuracy:", score

    cm = metrics.confusion_matrix(yTest, yPred)
    if plot_cm:
        plt.figure()
    util.plotConfusionMatrix(cm, classes=[chr(ord('a')+c) for c in np.unique(yTest)],title="Confusion Matrix", plot=plot_cm)
    if plot_cm:
        plt.show()

parser = argparse.ArgumentParser(description='Run SVM')
parser.add_argument('--ensemble-folder', default='classifiersForEnsemble', help='Ensemble folder')
parser.add_argument('--data-file', default='bcmnw.npz', help='Data file')
parser.add_argument('--majority', action='store_true', help='Use majority voting')
parser.add_argument('--leave-out', type=int, default=3, help='Index of person to leave out')
parser.add_argument('--disable-cm', dest='plot_cm', action='store_false', help='Do not plot a confusion matrix')

if __name__ == '__main__':
    namespace = parser.parse_args()
    args = vars(namespace)
    main(**args)
