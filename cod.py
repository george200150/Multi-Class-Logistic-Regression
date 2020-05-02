'''
Created on 2 mai 2020

@author: George
'''
from utils import load, plotLoss, plotModels, plotPredictions


xTrain, yTrain, xTest, yTest = load()



from regresie import LogisticRegression, LogisticBatchRegression
#from sklearn.linear_model import LogisticRegression
# model initialisation
model = LogisticBatchRegression()
#model = LogisticRegression()

# train the model (fit in on the training data)

#model.fit(xTrain, yTrain)
thetas, classes, costs = model.fit(xTrain[:,:2], yTrain)

# parameters of the liniar regressor
#w0, w1, w2 = model.intercept_, model.coef_[0], model.coef_[1]

w0, w1, w2 = model.intercept_, model.coef_[0], model.coef_[1]
#w0 = []
#w1 = []
#w2 = []
#w3 = []
#w4 = []
#for zero,one,two,three,four in thetas: # thetas contine tupluri a cate 5 elemente (wi, i=0..4), reprezentand pondeerile modelelor 1 vs all
#    w0.append(zero)
#    w1.append(one)
#    w2.append(two)
#    w3.append(three)
#    w4.append(four)

print('classification model: y(feature1, feature2) = ', w0, ' + ', w1, ' * feature1 + ', w2, ' * feature2')
#print('classification model: y(feature1, feature2, feature3, feature4) = ', w0, ' + ', w1, ' * feature2 + ', w2, ' * feature3 + ', w3, ' * feature4')


plotLoss(costs)

plotModels(thetas,xTrain,yTrain)


outputClasses = [0,1,2]


outputLabels = ['setosa','versicolor','virginica']


feature1test = [ex[0] for ex in xTest]
feature2test = [ex[1] for ex in xTest]


computedOutputs = model.predict(xTest[:,:2]) # only two features for better visualisation



plotPredictions(feature1test, feature2test, yTest, computedOutputs, "real test data", outputLabels, outputClasses)







model.fit(xTrain[:, :2], yTrain) # only two features (1 and 2)
print(f"Train Accuracy: {model.score(xTrain[:, :2], yTrain)}")
print(f"Test Accuracy: {model.score(xTest[:, :2], yTest)}")




model.fit(xTrain, yTrain) # all the features (1, 2, 3 and 4)
print(f"Train Accuracy: {model.score(xTrain, yTrain)}")
print(f"Test Accuracy: {model.score(xTest, yTest)}")
