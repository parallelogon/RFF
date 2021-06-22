import RFFRkhs as rf
import torch
import numpy as np
from sklearn.datasets import make_circles
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

m = 100
X, y = make_circles(m, factor=0.5, random_state=0, noise=0.05)
Y = np.reshape(y, (len(y), 1))
Y = 2*Y - 1  # Recoding for +(-)1 for better algorithm/optimization

X = (X - np.min(X))/(np.max(X)-np.min(X))
X = np.hstack((np.ones((X.shape[0], 1)), X))
X, X_test, Y, Y_test = train_test_split(X, Y, test_size=0.2)

c = rf.Clf(X,Y, lbda = .5)
c.Optimize(passes = 50, T = 100)
print("Training Accuracy: ", np.sum(Y*c.Predict(X) >= 0)/len(X))
print("Test Accuracy: ", np.sum(Y_test*c.Predict(X_test) >= 0)/len(X_test))
c.Show()
c.Fit()

#d = rf.hClf(X,Y,lbda = 1)

#rf.changeOfVariables(X,X)

#e = rf.HKGaussian(torch.tensor(X, dtype = torch.float), d.theta, d.omega)

#print(e)
#print(e.size())