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
