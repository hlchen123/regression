import numpy as np
class sgds(object):
    def __init__(self,X,Y):
        # preprocess the X and Y
        for i in range(len(X)):
            X[i] = X[i] + [1]
        self.X=X
        self.Y=Y
        n=len(self.X[0])

        self.w=np.random.randn(n) #initialize the w
        # self.w=np.ones(n)

# for w1,w2,w3...wn
    def one(self,j):#j the j of w

        m=len(self.X)
        g_=0
        for i in xrange(m):
            #g_=g_+2*(np.dot(self.w,self.X[i])-self.Y[i])*self.X[i][j]
            g_=g_+(self.Y[i]*self.X[i][j]-self.X[i][j]/(1+np.exp(-1*np.dot(self.w,self.X[i]))))
               #-2*lamda*self.w[j]/np.power(np.dot(self.w,self.w),2)
        return g_
    def fit(self,k=100,alpha=0.1):# k:the number of cycles
        n = len(self.X[0])
        for j in range(1, k):
            for i in range(n):
                if np.abs(self.one(i)) > 1e-10:
                    self.w[i] = self.w[i] +alpha * self.one(i)
        return self.w
# predict the category and probability
    def predict_proba(self,X):
        for i in range(len(X)):
            X[i] = X[i] + [1]
        n=len(X)
        y_pred_proba=[]
        for i in xrange(n):
            if self.sigmoid(X[i])>=0.5:
                y_pred_proba.append((1,self.sigmoid(X[i])))
            else:
                y_pred_proba.append((0,self.sigmoid(X[i])))
        return y_pred_proba
    def sigmoid(self,x):
        return 1/(1+np.exp(-1*np.dot(self.w,x)))


