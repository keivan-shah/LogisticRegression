
import numpy as np
import logging
import json
from utility import * #custom methods for data cleaning

FILE_NAME_TRAIN = 'train.csv' #replace this file name with the train file
FILE_NAME_TEST = 'test.csv' #replace
ALPHA = 1.6
EPOCHS = 50000
MODEL_FILE = 'models/model'
train_flag = True

logging.basicConfig(filename='output.log',level=logging.DEBUG)

np.set_printoptions(suppress=True)
#############################################################################################
################################ write the functions here ###################################
#############################################################################################
# this function appends 1 to the start of the input X and returns the new array
def appendIntercept(X):
    size = X.shape[0]
    o = np.ones([size,1])
    return np.hstack((o,X))

# intitial guess of parameters (intialize all to zero)
# this func takes the number of parameters that is to be fitted and returns a vector of zeros
def initialGuess(n_thetas):
    return np.zeros(n_thetas)

def logistic_func(prediction):
    return 1/(1+np.exp(-prediction))

def train(theta, X, y, model):
    m = len(y)
    for i in range(0,EPOCHS):
        y_predict = predict(X,theta)
        grad = calcGradients(X,y,y_predict,m)
        theta = makeGradientUpdate(theta,grad)
    model['theta'] = list(theta)
    return model

def costFunc(m,y,y_predicted):
    cost = y-y_predicted
    cost = np.square(cost)
    cost = np.sum(cost)
    return cost/(2*m)

def calcGradients(X,y,y_predicted,m):
    return (np.dot((y_predicted-y),X)/m)

# this function will update the theta and return it
def makeGradientUpdate(theta, grads):
    return theta-ALPHA*grads


# this function will take two paramets as the input
def predict(X,theta):
    return logistic_func(np.dot(X,theta))


######################## main function ###############################
def main():
    if(train_flag):
        model = {}
        X_df,y_df = loadData(FILE_NAME_TRAIN)
        X,y, model = normalizeData(X_df, y_df, model)
        X = appendIntercept(X)
        theta = initialGuess(X.shape[1])
        model = train(theta, X, y, model)

        # Training Accuracy
        print("Training Data Accuracy")
        accuracy(X,y,model)

        X_test,y_test = loadData(FILE_NAME_TEST)
        X_test,y_test = normalizeTestData(X_test, y_test, model)
        X_test = appendIntercept(X_test)

        # Testing Accuracy
        print("Testing Data Accuracy")
        accuracy(X_test,y_test,model)
        with open(MODEL_FILE,'w') as f:
            f.write(json.dumps(model))

if __name__ == '__main__':
    main()
