from sklearn.linear_model import LogisticRegression
clf=LogisticRegression()
X, Y = [[1, 2],[1,0], [2,3], [2,1]], [1, 0, 1, 0]
clf.fit(X,Y)
print clf.predict(X),clf.coef_

from sklearn.datasets import load_iris
clf_load=load_iris()
clf_load['data']