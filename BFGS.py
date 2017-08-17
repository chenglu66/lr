# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 16:52:22 2017

@author: Lenovo-Y430p
"""
from __future__ import division
from numpy import *
import matplotlib.pyplot as plt
def loadDataSet():
    dataMat = []; labelMat = []
    try:
        fr = open('E:\下载\machine-learning-ex-master\machine-learning-ex2\ex2\ex2data1.txt')
    except IOError:
        print("请检查您的路径")
    else:
        for line in fr.readlines():
            lineArr = line.strip().split(',')
            dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
            labelMat.append(int(lineArr[2]))
        fr.close()
        return dataMat,labelMat

def function(x):
    return x**2
def sigmoid(inX):
    return 1.0/(1+exp(-inX))

def function_deriv(g):
    #convert to NumPy matrix
    dataMatIn,classLabels=loadDataSet()
    dataMatrix = mat(dataMatIn) #convert to NumPy matrix
    labelMat = mat(classLabels).transpose()
    
    h = sigmoid(dataMatrix*g) #matrix mult
    error = (labelMat - h)#vector subtraction
    g=dataMatrix.T*error#matrix mult
    return g

#Identity Matrix
I = array( [ [1,0,0],
                [0,1,0],[0,0,1] ])

#Hessian Matrix set to Identity Matrix
B = I

#Hessian Matrix Inverse set to B
B_inv = B

#Pick initial point x_0
def plotBestFit(weights):
    import matplotlib.pyplot as plt
    dataMat,labelMat=loadDataSet()
    dataArr = array(dataMat)
    n = shape(dataArr)[0] 
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    for i in range(n):
        if int(labelMat[i])== 1:
            xcord1.append(dataArr[i,1]); ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1]); ycord2.append(dataArr[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = arange(20, 120, 10)
    y = (-weights[0]-weights[1]*x)/weights[2]+120
    y=y.tolist()
    ax.plot(x, y[0])
    plt.xlabel('X1'); plt.ylabel('X2');
    plt.show()
x_k = array([[1],[1],[1]])
for iterations in range(100):
    print ("Iteration ", iterations, ": ")
    #Obtain a direction p_k by solving B_k*p_k = -function_deriv(x_k)
    l=function_deriv(x_k)
    p_k = -B_inv.dot(l)
    #Perform line search to find acceptable stepsize alpha_k such that x_k+1 = x_k + alpha_k*p_k
    #-------- "min s |--> f( x_0 + s*p_0)"
    #--------  --> Newton(once)
    #------------- s_n+1 = s_n - g(s_n)/g'(s_n)
    #------------- s_0 = 0 (initial s_0 condition, could be anything)
    g = -function_deriv(x_k)
    print ("g", g)
    #Calculate step size alpha
    
    alpha_k = 200/(iterations+1)
    print ("alpha_k", alpha_k)
    #Set s_k = alpha_k*p_k
    s_k = alpha_k*p_k.T
    print ("s_k", s_k)
    #Set x_k+1 = x_k + alpha_k*p_k
    x_k1 = x_k-alpha_k*p_k
    print ("x_k" ,x_k,"x_k1", x_k1)
    #Set y_k = f'(x_k+1) - f'(x_k)
    print(function_deriv(x_k1))
    y_k = function_deriv(x_k1) - g
    print ("y_k", y_k)
    #Calculate B_k+1
    #B_k1 = B + ((y_k.dot(y_k.T))/(y_k.T.dot(s_k))) - ((B.dot(s_k.dot(s_k.T.dot(B))))/(s_k.T.dot(B.dot(s_k))))
    #Set new x_k
    x_k = x_k1
    #Compute new B_k inverse
    a = I - ((s_k.T.dot(y_k.T))/(y_k.T.dot(s_k.T)))
    b = I - ((y_k.dot(s_k))/(y_k.T.dot(s_k.T)))
    B_inv = a.dot(B_inv).dot(b) + ((s_k.T.dot(s_k))/(y_k.T.dot(s_k.T)))
    #B = B_k1
    print( "-------------------------------")
plotBestFit(x_k1)
