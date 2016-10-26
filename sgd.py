import numpy as np
class sgds(object):
    def __init__(self,X,Y):
        for i in range(len(X)):
            X[i] = X[i] + [1]
        self.X=X
        self.Y=Y
        n=len(self.X[0])
        #preprocess the X and Y
        self.w=np.random.random(size=n) #initialize the w
# for w1,w2,w3...wn
    def one(self,j):#j the j of w

        m=len(self.X)
        g_=0
        for i in xrange(m):
            g_=g_+2*(np.dot(self.w,self.X[i])-self.Y[i])*self.X[i][j]
        return g_
    def getanswers(self,k,lamda=0.1):# k:the number of cycles
        n = len(self.X[0])
        for j in range(1, k):
            for i in range(n):
                if np.abs(self.one(i)) > 1e-9:
                    self.w[i] = self.w[i] - lamda * self.one(i)
        return self.w