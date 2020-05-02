'''
Created on 2 mai 2020

@author: George
'''
import matplotlib.pyplot as plt
import numpy as np
import math
import pandas as pd

plt.rcParams["figure.figsize"] = (6, 4) # (w, h)
plt.rcParams["figure.dpi"] = 200
np.random.seed(1)



def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def plotSigmoid():
    x = np.linspace(-10, 10, 200)
    plt.plot(x, sigmoid(x))
    plt.axvline(x=0, color='k', linestyle='--');
    plt.title("Sigmoid");
    plt.show()



def plotCrossValidation():
    h = np.linspace(0, 1)[1:-1]
    for y in [0, 1]:
        plt.plot(h, -y * np.log(h) - (1 - y) * np.log(1 - h), label=f"y={y}")
    plt.title("Cross Entropy Loss") 
    plt.xlabel('$h_ {\\theta}(x)$'); plt.ylabel('$J(\\theta)$')
    plt.legend();
    plt.show()



def cost(theta, x, y):
    h = sigmoid(x @ theta)
    m = len(y)
    cost = softmax(h,y)
    #cost = mae(h,y)
    #cost = rmse(h,y)
    #cost = huber(h,y,delta=0.5)
    grad = 1 / m * ((y - h) @ x)
    return cost, grad

def softmax(x, y):
    m = len(y)
    cost = 1 / m * np.sum(-y * np.log(x) - (1 - y) * np.log(1 - x))
    return cost

# TODO: MAE loss
def mae(true, pred):
    errorL1 = np.sum(np.abs(r - c) for r, c in zip(true, pred)) / len(true)
    return errorL1

# TODO: RMSE loss
def rmse(true, pred):
    errorL2 = math.sqrt(np.sum((r - c) ** 2 for r, c in zip(true, pred)) / len(true))
    return errorL2

# TODO: huber loss
def huber(true, pred, delta):
    loss = np.where(np.abs(true-pred) < delta , 0.5*((true-pred)**2), delta*np.abs(true - pred) - 0.5*(delta**2))
    return np.sum(loss)

# TODO: log cosh loss
def logcosh(true, pred):
    loss = np.log(np.cosh(pred - true))
    return np.sum(loss)



def load():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    dataFrame = pd.read_csv(url, header=None, names=[
        "Sepal length (cm)", 
        "Sepal width (cm)", 
        "Petal length (cm)",
        "Petal width (cm)",
        "Species"
        ])
    dataFrame.head()
    
    dataFrame['Species'] = dataFrame['Species'].astype('category').cat.codes
    data = np.array(dataFrame)
    np.random.shuffle(data)
    num_train = int(.8 * len(data))  # 80/20 train/test split
    x_train, y_train = data[:num_train, :-1], data[:num_train, -1]
    x_test, y_test = data[num_train:, :-1], data[num_train:, -1]
    
    plt.scatter(x_train[:,0], x_train[:, 1], c=y_train, alpha=0.5)
    plt.xlabel("Sepal Length (cm)"); plt.ylabel("Sepal Width (cm)");
    plt.show()
    
    return x_train, y_train, x_test, y_test



def plotLoss(costs):
    plt.plot(costs)
    plt.xlabel('Number Epochs')
    plt.ylabel('Cost')
    plt.show()


def plotModels(thetas, X, y):
    plt.scatter(X[:,0], X[:, 1], c=y, alpha=0.5)
    plt.xlabel("Sepal Length (cm)")
    plt.ylabel("Sepal Width (cm)")
    for theta in [thetas[0],thetas[2]]:
        j = np.array([X[:, 0].min(), X[:, 0].max()])
        k = -(j * theta[1] + theta[0]) / theta[2]
        plt.plot(j, k, color='k', linestyle="--")
    plt.show()


def plotPredictions(feature1, feature2, realOutputs, computedOutputs, title, labelNames, outputs):
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
    plt.xlabel('Sepal length (cm)')
    plt.ylabel('Sepal width (cm)') # cannot plot 4 dimensions => use PCA in order to reduce dimensionality for better comprehension
    plt.legend()
    plt.title(title)
    plt.show()


if __name__ == '__main__':
    plotSigmoid()
    plotCrossValidation()