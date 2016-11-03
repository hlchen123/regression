import sgd
import newton
import time
import sgd_logistic
import pylab as pl
import numpy as np
from sklearn.datasets import load_iris
iris=load_iris()
X, Y = [[1, 2],[1,0], [2,3], [2,1]], [1, 0, 1, 0]
begin = time.time()
clf_sgd = sgd_logistic.sgds(X, Y)
#1/0
print clf_sgd.fit(20000,0.2)

print clf_sgd.predict_proba([[1, 2],[1,0], [2,3], [2,1]])
