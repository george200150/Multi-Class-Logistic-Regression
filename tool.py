'''
Created on 2 mai 2020

@author: George
'''
import warnings
warnings.filterwarnings("ignore")

'''import matplotlib.pyplot as plt
from sklearn import datasets


iris = datasets.load_iris()


X = iris.data[:, :2]
y = (iris.target != 0) * 1



plt.figure(figsize=(10, 6))
plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color='b', label='0')
plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='r', label='1')
plt.legend();
plt.show()


from sklearn.linear_model import LogisticRegression

model = LogisticRegression()


model.fit(X, y)

preds = model.predict(X)
(preds == y).mean()

print(model.intercept_, model.coef_)'''




"""from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()
inputs = data['data']
outputs = data['target']
outputNames = data['target_names']
featureNames = list(data['feature_names'])
feature1 = [feat[featureNames.index('mean radius')] for feat in inputs]
feature2 = [feat[featureNames.index('mean texture')] for feat in inputs]
inputs = [[feat[featureNames.index('mean radius')], feat[featureNames.index('mean texture')]] for feat in inputs]

import matplotlib.pyplot as plt
labels = set(outputs)
noData = len(inputs)
for crtLabel in labels:
    x = [feature1[i] for i in range(noData) if outputs[i] == crtLabel ]
    y = [feature2[i] for i in range(noData) if outputs[i] == crtLabel ]
    plt.scatter(x, y, label = outputNames[crtLabel])
plt.xlabel('mean radius')
plt.ylabel('mean texture')
plt.legend()
plt.show()


from sklearn.preprocessing import StandardScaler

def normalisation(trainData, testData):
    scaler = StandardScaler()
    if not isinstance(trainData[0], list):
        #encode each sample into a list
        trainData = [[d] for d in trainData]
        testData = [[d] for d in testData]
        
        scaler.fit(trainData)  #  fit only on training data
        normalisedTrainData = scaler.transform(trainData) # apply same transformation to train data
        normalisedTestData = scaler.transform(testData)  # apply same transformation to test data
        
        #decode from list to raw values
        normalisedTrainData = [el[0] for el in normalisedTrainData]
        normalisedTestData = [el[0] for el in normalisedTestData]
    else:
        scaler.fit(trainData)  #  fit only on training data
        normalisedTrainData = scaler.transform(trainData) # apply same transformation to train data
        normalisedTestData = scaler.transform(testData)  # apply same transformation to test data
    return normalisedTrainData, normalisedTestData


def plotClassificationData(feature1, feature2, outputs, title = None):
    labels = set(outputs)
    noData = len(feature1)
    for crtLabel in labels:
        x = [feature1[i] for i in range(noData) if outputs[i] == crtLabel ]
        y = [feature2[i] for i in range(noData) if outputs[i] == crtLabel ]
        plt.scatter(x, y, label = outputNames[crtLabel])
    plt.xlabel('mean radius')
    plt.ylabel('mean texture')
    plt.legend()
    plt.title(title)
    plt.show()

# step2: impartire pe train si test
# step3: normalizare 
import numpy as np

# split data into train and test subsets
np.random.seed(5)
indexes = [i for i in range(len(inputs))]
trainSample = np.random.choice(indexes, int(0.8 * len(inputs)), replace = False)
testSample = [i for i in indexes  if not i in trainSample]

trainInputs = [inputs[i] for i in trainSample]
trainOutputs = [outputs[i] for i in trainSample]
testInputs = [inputs[i] for i in testSample]
testOutputs = [outputs[i] for i in testSample]

#normalise the features
trainInputs, testInputs = normalisation(trainInputs, testInputs)

#plot the normalised data
feature1train = [ex[0] for ex in trainInputs]
feature2train = [ex[1] for ex in trainInputs]
feature1test = [ex[0] for ex in testInputs]
feature2test = [ex[1] for ex in testInputs]  

plotClassificationData(feature1train, feature2train, trainOutputs, 'normalised train data')




from sklearn.linear_model import LogisticRegression

# model initialisation
classifier = LogisticRegression() 

# train the classifier (fit in on the training data)
classifier.fit(trainInputs, trainOutputs)
# parameters of the liniar regressor
#w0, w1, w2 = classifier.intercept_, classifier.coef_[0], classifier.coef_[1]
w0, w1 = classifier.intercept_, classifier.coef_[0]
#print('classification model: y(feat1, feat2) = ', w0, ' + ', w1, ' * feat1 + ', w2, ' * feat2')
print('classification model: y(feat1, feat2) = ', w0, ' + ', w1, ' * feat1')

computedTestOutputs = classifier.predict(testInputs)

def plotPredictions(feature1, feature2, realOutputs, computedOutputs, title, labelNames):
    labels = list(set(outputs))
    noData = len(feature1)
    for crtLabel in labels:
        x = [feature1[i] for i in range(noData) if realOutputs[i] == crtLabel and computedOutputs[i] == crtLabel ]
        y = [feature2[i] for i in range(noData) if realOutputs[i] == crtLabel and computedOutputs[i] == crtLabel]
        plt.scatter(x, y, label = labelNames[crtLabel] + ' (correct)')
    for crtLabel in labels:
        x = [feature1[i] for i in range(noData) if realOutputs[i] == crtLabel and computedOutputs[i] != crtLabel ]
        y = [feature2[i] for i in range(noData) if realOutputs[i] == crtLabel and computedOutputs[i] != crtLabel]
        plt.scatter(x, y, label = labelNames[crtLabel] + ' (incorrect)')
    plt.xlabel('mean radius')
    plt.ylabel('mean texture')
    plt.legend()
    plt.title(title)
    plt.show()

plotPredictions(feature1test, feature2test, testOutputs, computedTestOutputs, "real test data", outputNames)


# evalaute the classifier performance
# compute the differences between the predictions and real outputs
# print("acc score: ", classifier.score(testInputs, testOutputs))
error = 0.0
for t1, t2 in zip(computedTestOutputs, testOutputs):
    if (t1 != t2):
        error += 1
error = error / len(testOutputs)
print("classification error (manual): ", error)

from sklearn.metrics import accuracy_score
error = 1 - accuracy_score(testOutputs, computedTestOutputs)
print("classification error (tool): ", error)
"""











from utils import load
from sklearn.linear_model import LogisticRegression
import numpy as np
from utils import plotPredictions

xTrain, yTrain, xTest, yTest = load()
model = LogisticRegression()
model.fit(xTrain[:,:2], yTrain)
w0, w1, w2 = model.intercept_, model.coef_[0], model.coef_[1]
thetas = np.asarray([w0, w1, w2])


outputs = [0,1,2]
outputNames = ['setosa','versicolor','virginica']



feature1test = [ex[0] for ex in xTest]
feature2test = [ex[1] for ex in xTest]


computedTestOutputs = model.predict(xTest[:,:2]) # only two features for better visualisation

plotPredictions(feature1test, feature2test, yTest, computedTestOutputs, "test data", outputNames, outputs)

error = 0.0
for t1, t2 in zip(computedTestOutputs, yTest):
    if (t1 != t2):
        error += 1
error = error / len(yTest)
print("classification error (manual): ", error)

from sklearn.metrics import accuracy_score
error = 1 - accuracy_score(yTest, computedTestOutputs)
print("classification error (tool): ", error)


print("Train data accuracy: ", (model.predict(xTrain[:,:2]) == yTrain).mean())
print("Test data accuracy: ", (model.predict(xTest[:,:2]) == yTest).mean())


model.fit(xTrain, yTrain)
print("Train data accuracy: ", (model.predict(xTrain) == yTrain).mean())
print("Test data accuracy: ", (model.predict(xTest) == yTest).mean())
