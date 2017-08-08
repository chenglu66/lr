from numpy import *
import matplotlib.pyplot as plt
import sys
sys.path.append(r'C:\Users\Lenovo-Y430p\Documents\Python Scripts')
#from L_BFGS import *
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

def sigmoid(inX):
    return 1.0/(1+exp(-inX))

def gradAscent(dataMatIn, classLabels):
    dataMatrix = mat(dataMatIn)             #convert to NumPy matrix
    labelMat = mat(classLabels).transpose() #convert to NumPy matrix
    m,n = shape(dataMatrix)
    maxCycles = 5009
    weights = ones((n,1))
    for k in range(maxCycles):#heavy on matrix operations
        alpha = 0.9/(k+1)
        h = sigmoid(dataMatrix*weights)     #matrix mult
        error = (labelMat - h)              #vector subtraction
        weights = weights + alpha * dataMatrix.T*error #matrix mult
    return weights.T

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
    x = arange(20, 120, 0.1)
    y = (-weights[0]-weights[1]*x)/weights[2]
    ax.plot(x, y)
    plt.xlabel('X1'); plt.ylabel('X2');
    plt.show()

def stocGradAscent0(dataMatrix, classLabels):
    m,n = shape(dataMatrix)
    alpha = 0.01
    weights = ones(n)   #initialize to all ones
    for i in range(m):
        h = sigmoid(sum(dataMatrix[i]*weights))
        error = classLabels[i] - h
        weights = weights + alpha * error * dataMatrix[i]
    return weights

def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    m,n = shape(dataMatrix)
    weights = ones(n)   #initialize to all ones
    for j in range(numIter):
        dataIndex = [ 0 for i in range(m)]
        for i in range(m):
            alpha = 4/(1.0+j+i)+0.0001    #apha decreases with iteration, does not 
            randIndex = int(random.uniform(0,len(dataIndex)))#go to 0 because of the constant
            h = sigmoid(sum(dataMatrix[randIndex]*weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            del(dataIndex[randIndex])
    return weights

def classifyVector(inX, weights):
    prob = sigmoid(sum(inX*weights))
    if prob > 0.5: return 1.0
    else: return 0.0

def colicTest():
    frTrain = open('E:\下载\machine-learning-ex-master\machine-learning-ex2\ex2\ex2data1.txt'); frTest = open('E:\下载\machine-learning-ex-master\machine-learning-ex2\ex2\ex2data2.txt')
    trainingSet = []; trainingLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split(',')
        lineArr =[1.0]
        for i in range(2):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[-1]))
    trainWeights = stocGradAscent1(array(trainingSet), trainingLabels, 1000)
    #trainWeights=gradAscent(trainingSet, trainingLabels)
    #t=[trainWeights[0,0],trainWeights[0,1],trainWeights[0,2]]
    plotBestFit(trainWeights)
    return 
    errorCount = 0; numTestVec = 0.0
    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr =[]
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(array(lineArr), trainWeights))!= int(currLine[21]):
            errorCount += 1
    errorRate = (float(errorCount)/numTestVec)
    print ("the error rate of this test is: %f" % errorRate)
    return errorRate
def multiTest():
    numTests = 10; errorSum=0.0
    for k in range(numTests):
        errorSum += colicTest()
    print ("after %d iterations the average error rate is: %f" % (numTests, errorSum/float(numTests)))
def main():
    colicTest()
    #multiTest()
    #plotBestFit(weights)
def test():
    dataMatIn,classLabels=loadDataSet()
    dataMatrix = mat(dataMatIn) #convert to NumPy matrix
    labelMat = mat(classLabels).transpose() #convert to NumPy matrix
    M,n=shape(dataMatrix)
    x=ones((n,1))
    fx=dot(dataMatrix,x)
    param = lbfgs_parameters(new_Evaluate, progress)
    lb = lbfgs(n, x, fx, param)
    ret = lb.do_lbfgs()
    plotBestFit(x)
    

def new_Evaluate(w, g, n, step):
    dataMatIn,classLabels=loadDataSet()
    dataMatrix = mat(dataMatIn) #convert to NumPy matrix
    labelMat = mat(classLabels).transpose() #convert to NumPy matrix
    h = sigmoid(dataMatrix*w) #matrix mult
    error = (labelMat - h)#vector subtraction
    g=dataMatrix.T*error#matrix mult
    fx=dot(dataMatrix,w)
    return fx,g
def progress(x, g, fx, xnorm, gnorm, step, n, k, ls):
    pass
if __name__=='__main__':
    main()  
