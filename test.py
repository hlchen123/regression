import sgd
import newton
import time
import sgd_logistic
import pylab as pl
import numpy as np

X, Y = [[1, 1], [1, 2], [2, 1], [1, 1]], [8, 12, 11, 9]
begin = time.time()
clf_sgd = sgd.sgds(X, Y)

print clf_sgd.getanswers(1000), time.time() - begin

# X,Y=[[1,1],[1,2],[2,1]],[8,12,11]
X, Y = [[1, 1], [1, 2], [2, 1], [1, 1]], [8, 12, 11, 9]
begin = time.time()
clf_newton = newton.newTon(X, Y)
print clf_newton.getanswers(1000, lamda=0.1), time.time() - begin
X, Y = [[1, 1], [1, 2], [2, 1]], [8, 12, 11]