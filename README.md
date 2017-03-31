# aslRecognition

### Performance on bcmnw (testing on person C):

- SVC(kernel=linear, C=1), color: 0.51
- SVC(kernel=linear, C=1), depth: 0.88
- SVC(kernel=linear, C=1), color and depth: 0.85

##### Parameters chosen by leave one person out cv:

- SVC(kernel=linear, C=0.001, best for color), color: 0.47
- SVC(kernel=linear, C=0.01, best for depth), depth: 0.89
- SVC(kernel=linear, C=0.001, best for color and depth), color and depth: 0.86
- SVC(kernel=rbf, C=10, gamma=0.001, best for color), color: 0.37
- SVC(kernel=rbf, C=10, gamma=0.001, best for depth), depth: 0.91
- SVC(kernel=rbf, C=100, gamma=0.0001, best for color and depth), color and depth: 0.88

### Performance on overfeat_bcmnw (testing on person C):

##### Parameters chosen by leave one person out cv:

- SVC(kernel=linear, C=0.01, best for color), color: 0.67
- SVC(kernel=linear, C=0.01, best for depth), depth: 0.87
- SVC(kernel=linear, C=0.01, best for color and depth), color and depth: 0.84 (interestingly, the cv accuracies were much higher (0.92) than for the other settings)
- SVC(kernel=rbf, C=10, gamma=0.0001, best for color), color: 0.67
- SVC(kernel=rbf, C=10, gamma=0.0001, best for depth), depth: 0.88
- SVC(kernel=rbf, C=10, gamma=0.00001, best for color and depth), color and depth: 0.84

### Performance with additional classifiers and features (bcmnw)
- KNN(k=3) depth (scaled 0.2): 0.86
- KNN(k=9) gabor: 0.84
- KNN(k=3) depth (0.2 scale) + gabor: 0.85
- SVC(kernel=linear, C=0.1) gabor: 0.88
- SVC(kernel=linear, C=0.1) depth (0.2 scale) + gabor: 0.95
- SVC(kernel=rbf, C=10, gamma=0.01) depth (0.2 scale) + gabor: 0.96
- LogReg(C=1.0,multinomial) depth (0.2 scale): 0.80
- LogReg(C=1.0,multinomial) gabor: 0.86
- LogReg(C=1.0,multinomial) gabor + depth (0.2 scale): 0.91
- RF('max_features': 0.28704608069377779, 'n_estimators': 19, 'max_depth': 8) overfeat: 0.76
- RF('max_features': 0.46912474635711165, 'n_estimators': 15, 'max_depth': 6) gabor: 0.76
- RF('max_features': 0.25839424397640276, 'n_estimators': 17, 'max_depth': 8) depth (0.2 scale): 0.53

### Ensemble Performance

So far only found one ensemble that improves at all:
- LogRegOverfeat, SVMLinearDepth03Gabor, SVMRBFLinearDepth03Gabor: 0.963

### Performance on all letters
- SVC(kernel=rbf, C = 100.0, gamma = 0.01) depth@0.3 gabor: 0.64
- SVC(kernel=linear, C = 10) depth@0.3 gabor: 0.65
- SVC(kernel=linear, C = 0.01) overfeat[depth]: 0.69
- SVC(kernel=rbf, C = 100, gamma = 0.0001) overfeat[depth]: 0.70
- Ensemble of
    + SVC(kernel=linear, C = 10) depth@0.3 gabor
    + SVC(kernel=linear, C = 0.01) overfeat[depth]
  had performance: 0.73
- Ensemble of
    + SVC(kernel=rbf, C = 100.0, gamma = 0.01) depth@0.3 gabor: 0.64
    + SVC(kernel=linear, C = 10) depth@0.3 gabor: 0.65
    + SVC(kernel=linear, C = 0.01) overfeat[depth]: 0.69
    + SVC(kernel=rbf, C = 100, gamma = 0.0001) overfeat[depth]: 0.70
  had performance: 0.74