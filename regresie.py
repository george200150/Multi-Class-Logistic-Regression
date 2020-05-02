'''
Created on 2 mai 2020

@author: George
'''
import numpy as np
from utils import cost, sigmoid



class LogisticBatchRegression:
    def __init__(self):
        self.thetas = []
        self.classes = []
        self.costs = []
        self.intercept_ = None
        self.coef_ = []
    
    def fit(self, x, y, max_iter=5000, alpha=0.1):
        x = np.insert(x, 0, 1, axis=1)
        self.thetas = []
        self.classes = np.unique(y) # get the unique labels (0,1,2 for the three species of flowers)
        self.costs = np.zeros(max_iter)
    
        for c in self.classes:
            # one vs. rest binary classification
            
            theta = np.zeros(x.shape[1])
            for epoch in range(max_iter):
                batchSize = len(y)//10 # variable parameter - you call it
                m = len(y)
                indices = np.random.permutation(m) # shuffle the data to avoid cycles
                x = x[indices]
                y = y[indices]
                for i in range(0,m,batchSize): # split the data in batches
                    x_i = x[i:i+batchSize]
                    y_i = y[i:i+batchSize]
                    binary_y_batch = np.where(y_i == c, 1, 0) # create an array containing only the labeled outputs of the current class
                    _, grad = cost(theta, x_i, binary_y_batch)
                    theta += alpha * grad
                binary_y_complete = np.where(y == c, 1, 0)
                self.costs[epoch], _ = cost(theta, x, binary_y_complete)
            
            self.thetas.append(theta)
        
        self.intercept_ = self.thetas[0]
        self.coef_ = self.thetas[1:]
        return self.thetas, self.classes, self.costs
    
    
    def predict(self, x):
        x = np.insert(x, 0, 1, axis=1)
        preds = [np.argmax([sigmoid(xi @ theta) for theta in self.thetas]) for xi in x]
        return [self.classes[p] for p in preds]


    def score(self, x, y):
        return (self.predict(x) == y).mean()









class LogisticRegression:
    def __init__(self):
        self.thetas = []
        self.classes = []
        self.costs = []
        self.intercept_ = None
        self.coef_ = []
    
    def fit(self, x, y, max_iter=5000, alpha=0.1):
        x = np.insert(x, 0, 1, axis=1)
        self.thetas = []
        self.classes = np.unique(y)
        self.costs = np.zeros(max_iter)
    
        for c in self.classes:
            # one vs. rest binary classification
            binary_y = np.where(y == c, 1, 0)
            
            theta = np.zeros(x.shape[1])
            for epoch in range(max_iter):
                self.costs[epoch], grad = cost(theta, x, binary_y)
                theta += alpha * grad
                
            self.thetas.append(theta)
        
        self.intercept_ = self.thetas[0]
        self.coef_ = self.thetas[1:]
        return self.thetas, self.classes, self.costs
    
    
    def predict(self, x):
        x = np.insert(x, 0, 1, axis=1)
        preds = [np.argmax([sigmoid(xi @ theta) for theta in self.thetas]) for xi in x] # am facut evaluarea ca la reteaua mea neuronala
        return [self.classes[p] for p in preds]


    def score(self, x, y):
        return (self.predict(x) == y).mean()




# Threshold variation in model performane: - optimizing model performance
#
# Back to our goal of classifying breast cancer in patients given certain statistics about the patient. 
# Our model has quite a low recall of 0.5 This means that our model has many false negatives: 
#    it did not predict cancer was present, while actually the patient did have cancer.
# Given our so-called domain knowledge, we might try to optimize for recall instead of accuracy: 
#    we would rather predict too many patients have cancer than too few.

# Deci, modificand threshold-ul pentru cat de critic este modelul, putem avea o confidenta mai mare in luarea unor decizii pentru o clasa, 
#    in defavoarea preciziei unei alte clase.

# Astfel, in anumite situatii este preferabil ca diagnosticul dat de model sa fie mai putin precis, fiind dat mai multor persoane decat
#    ar fi fost nevoie, dar avem o siguranta mai mare ca modelul nu a scapat din vedere un pacient care avea cancer si l-ar fi 
#    diagnosticat negativ, cand el era pozitiv. Pacientii falsi pozitivi nu au de suferit in urma acestui diagnostic incorect, ceea ce
#    ne permite sa constrangem modelul sa fie cat mai sensibil sa cele mai mici semne de cancer.

