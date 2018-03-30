# Support Vectors Machine from scratch
# Author: Khalil Khalil
# 1/18/2017



import numpy as np
from cvxopt import matrix ,solvers
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import svm

class SVM(object):
    def __init__(self,kernel="polynomial",c=1,gamma=1,d=1):
        self.kernel = kernel
        self.c = c
        self.gamma = gamma
        self.d = d
        self.lagrangeMultipliers = []
        self.lagrangeMultipliers_sv = []
        self.w = []
        self.b = 0
        self.support_vectors = []
        self.support_vectors_labels = []


    def gram_matrix(self,X):
        N = X.shape[0]
        D = X.shape[1]
        G = np.zeros((N,N))
        for i,x_i in enumerate(X):
            for j,x_j in enumerate(X):
                G[i,j] = self.kernelFun(x_i,x_j)
        return G

    def kernelFun(self,x_i,x_j):
        xi = np.array(x_i)
        xj = np.array(x_j)
        if self.kernel == "polynomial":
            return np.power(xi.T.dot(xj),self.d)
        elif self.kernel == "rbf":
            dist = np.linalg.norm(xi-xj)**2
            return np.exp(-self.gamma*dist)


    def fit(self,X,y):
        N = X.shape[0]
        D = X.shape[1]
        Gram = self.gram_matrix(X)
        # q^T*a-(1/2)*a^T*P*a
        q = matrix(-1*np.ones(N))
        Parray = np.zeros((N,N))
        for i in range(0,N):
            for j in range(0,N):
                Parray[i,j] = y[i]*y[j]*Gram[i,j]
        P = matrix(Parray)
        G1 = np.diag(np.ones(N))
        G2 = np.diag(-1*np.ones(N))
        G = matrix(np.vstack((G1,G2)))
        h1 = np.ones(N)*self.c
        h2 = np.zeros(N)
        h = matrix(np.hstack((h1,h2)))
        A = matrix(y,(1,N),'d')
        b = matrix(0.0)
        sol = solvers.qp(P,q,G,h,A,b)
        self.lagrangeMultipliers = np.ravel(sol["x"])
        support_vectors_indices = self.lagrangeMultipliers > 1e-03
        self.lagrangeMultipliers_sv = self.lagrangeMultipliers[support_vectors_indices]
        self.support_vectors = X[support_vectors_indices]
        self.support_vectors_labels = y[support_vectors_indices]
        support_vector_margin_indices = (self.lagrangeMultipliers > 1e-03) & (self.lagrangeMultipliers < self.c - 1e-05)
        support_vector_margin = X[support_vector_margin_indices]
        support_vector_margin_labels = y[support_vector_margin_indices]
        print("margin: ",self.lagrangeMultipliers[support_vector_margin_indices]," ", self.lagrangeMultipliers[support_vectors_indices])
        sum_n = 0
        for x_n,y_n in zip(support_vector_margin,support_vector_margin_labels):
            sum_m = 0
            for x_m,y_m, a_m in zip(self.support_vectors,self.support_vectors_labels,self.lagrangeMultipliers_sv):
                sum_m += a_m*y_m*self.kernelFun(x_n,x_m)
            sum_n += y_n - sum_m
        self.b = sum_n / support_vector_margin.shape[0]

    def predict(self,X):
         y = np.zeros(X.shape[0])
         for i in range(0, y.shape[0]):
             for a_n, y_n, x_n in zip(self.lagrangeMultipliers_sv, self.support_vectors_labels, self.support_vectors):
                 y[i] += a_n * y_n * self.kernelFun(X[i, :], x_n)
             y[i] += self.b
         return np.sign(y)





