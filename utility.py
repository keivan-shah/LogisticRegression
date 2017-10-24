import numpy as np
import pandas as pd #not of your use
import logging
import json


logging.basicConfig(filename='output.log',level=logging.DEBUG)


#utility functions
def loadData(file_name):
    df = pd.read_csv(file_name)
    logging.info("Number of data points in the data set "+str(len(df)))
    y_df = df['output']
    keys = ['model_rating', 'bought_at', 'months_used', 'issues_rating','company_rating','resale_value']
    X_df = df.get(keys)
    return X_df, y_df


def normalizeData(X_df, y_df, model):
    #save the scaling factors so that after prediction the value can be again rescaled
    model['input_scaling_factors'] = [list(X_df.mean()),list(X_df.std())]

    X = np.array((X_df-X_df.mean())/X_df.std())

    return X, y_df, model

def normalizeTestData(X_df, y_df, model):
    meanX = model['input_scaling_factors'][0]
    stdX = model['input_scaling_factors'][1]

    X = 1.0*(X_df - meanX)/stdX

    return X, y_df


def accuracy(X, y, model):

    y_predicted = predict(X,np.array(model['theta']))
    err = (np.sum(np.square(y_predicted - y)))/len(X)*100

    print("Accuracy associated with this model is "+str(100-err))

def logistic_func(prediction):
    return 1.0/(1.0+np.exp(-prediction))

def predict(X,theta):
    p = np.dot(X,theta)
    p = logistic_func(p)
    a = np.array([])
    s = ""
    for v in p:
        if v>0.5:
            a = np.append(a,[1])
        else:
            a = np.append(a,[0])
    return a
