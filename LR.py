# -*- coding: utf-8 -*-
"""
Created on Sat Jul 15 14:22:32 2017

@author: Lenovo-Y430p
"""
from numpy import *
import matplotlib.pyplot as plt
import math
'''
载入数据
'''
def loaddata(filename):
    fr=open(filename)
    datamat=[]
    label=[]
    for lines in fr.readlines():
        line=lines.split(',')
        s=list(map(float,line[:2]))
        s.insert(0,1)
        datamat.append(s)
        label.append(float(line[-1]))
    return mat(datamat),mat(label).T
'''
sigmod函数
'''
def sigmode(x):
    return 1.0/(1+exp(-x))
'''
批量梯度下降法
'''
def gradAscent(dataMatIn, labelMat):
    dataMatrix = mat(dataMatIn)             #convert to NumPy matrix
    m,n = shape(dataMatrix)
    alpha = 0.001
    maxCycles = 9000
    weights = ones((n,1))
    for k in range(maxCycles):              #heavy on matrix operations
        h = sigmode(dataMatrix*weights)     #matrix mult
        error = (labelMat - h)              #vector subtraction
        weights = weights + alpha * dataMatrix.transpose()* error #matrix mult
    return weights
'''
未加入正则的损失函数
'''
def cost(x,y,theta):
    f=np.dot(x, theta)
    f=sigmode(f)
    f1=dot(y,log(f))
    f2=dot(1-y,log(1-f))
    return f1+f2
'''
加入正则的的损失函数
'''
def costre(x,y,theta,c):
    f=dot(x, theta)
    f=sigmode(f)
    f1=dot(y,math.log(f))
    f2=dot(1-y,log(1-f))
    rem=c*sum(theta[1::]*theta[1::])
    return f1+f2+rem

'''
随机梯度下降法1
'''
def stocGradAscent1(dataMatrix, classLabels):
    m,n = shape(dataMatrix)
    alpha = 0.01
    weights = ones((n,1))   #initialize to all ones
    for i in range(m):
        h = sigmode(sum(dataMatrix[i]*(weights)))
        error = classLabels[i] - h
        weights = weights + alpha * error * dataMatrix[i]
        print(weights)
    return weights

'''
随机梯度下降法2
'''
def stocGradAscent2(x, y, theta, alpha): 
    numIter =1000
    
    m,n = shape(x)
    for j in range(numIter):
        dataIndex =list(range(m))
        for i in range(m):
            alpha = 4/(1.0+j+i)+0.0001 #apha decreases with iteration, does not 
            randIndex = int(random.uniform(0,len(dataIndex)))#go to 0 because of the constant
            h = sigmode(dot(x[randIndex,:],theta.T))
            error = y[randIndex] - h
            #print(x[randIndex,:])
            theta += (alpha * error * x[randIndex])
            print(theta)
            del(dataIndex[randIndex])
    return theta
if __name__=='__main__':
   x,y=loaddata('E:\下载\machine-learning-ex-master\machine-learning-ex2\ex2\ex2data1.txt')
   m,n = shape(x)
   theta= mat(ones(n))
   t=stocGradAscent2(x, y, theta, alpha=1)
   #t=stocGradAscent1(x, y,)
   #t=gradAscent(x, y)
   #print(weight)
   t=t.tolist()
   #print(t[0][2])
   temp=[]
   temp1=[]
   for i in range(m):
       if int(y[i])==1:
           temp.append(x[i,1:].tolist()[0])
       else:
           temp1.append(x[i,1:].tolist()[0])
   temp=mat(temp)
   temp1=mat(temp1)
   x1=arange(20,110,10)
   y=(float(t[0][0])+float(t[0][1])*x1)/(-float(t[0][2]))
   #y=(float(t[0][0])+float(t[1][0])*x1)/(-float(t[2][0]))
   fig=plt.figure()
   ax=fig.add_subplot(111)
   ax.scatter(temp[:,0],temp[:,1],c='r',marker=r'$\bigodot$',s=70)
   ax.scatter(temp1[:,0],temp1[:,1],c='b',marker='s',s=70)
   ax.plot(x1,y)
   plt.show()
   
   
