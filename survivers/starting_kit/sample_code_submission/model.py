'''
Sample predictive model.
You must supply at least 4 methods:
- fit: trains the model.
- predict: uses the model to perform predictions.
- save: saves the model.
- load: reloads the model.
'''
import numpy as np   # We recommend to use numpy arrays
from os.path import isfile
from sklearn.base import BaseEstimator
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle 

from sklearn.tree import DecisionTreeRegressor                  #
from sklearn.naive_bayes import GaussianNB                      #
from sklearn.linear_model import LinearRegression               # Les différentes régressions que  
from sklearn.neighbors.nearest_centroid import NearestCentroid  # nous avons utilisé
from sklearn.ensemble import RandomForestClassifier             #
from sklearn.ensemble import GradientBoostingRegressor          #

from sklearn.ensemble import BaggingRegressor       #
from sklearn.ensemble import BaggingClassifier      # Pour faire voter 
from sklearn.ensemble import VotingClassifier       # les regressions
from sklearn.pipeline import Pipeline               #

from sklearn.decomposition import PCA
    ############ nouvelle idée ############
class Preprocessor(BaseEstimator): # comme dans le tp 2
    def __init__(self):
        self.transformer = PCA(n_components=10)

    def fit(self, X, y=None):
        return self.transformer.fit(X, y)

    def fit_transform(self, X, y=None):
        return self.transformer.fit_transform(X)

    def transform(self, X, y=None):
        return self.transformer.transform(X)
    #############################
class model(BaseEstimator):
    def __init__(self, choice):
        '''
        This constructor is supposed to initialize data members.
        Use triple quotes for function documentation.
        '''
        self.num_train_samples=0
        self.num_feat=1
        self.num_labels=1
        self.is_trained=False

        ''' Baseline decision tree : 0.736285037 '''
        if choice == 0 :
            self.baseline_clf = DecisionTreeRegressor(max_depth=4)
     
        ''' Naivebayes : score 0.2084484036 '''
        if choice == 1 :
            self.baseline_clf = GaussianNB()
        
        ''' linaire regression : score 0.683989487 '''
        if choice == 2 :
            self.baseline_clf = LinearRegression()
        
        ''' random forest : score 0.4795732662 '''
        if choice == 3 :
            self.baseline_clf = RandomForestClassifier()
        
        ''' nearest neighbors : score 0.2249313468 '''
        if choice == 4 :
            self.baseline_clf = NearestCentroid()
            
        ''' GradientBoostingRegressor : score 0.7800 '''
        if choice == 5 :
            self.baseline_clf = GradientBoostingRegressor()  # notre meilleur régression 

# Fin de partie régression

    def fit(self, X, y):
        '''
        This function should train the model parameters.
        Here we do nothing in this example...
        Args:
            X: Training data matrix of dim num_train_samples * num_feat.
            y: Training label matrix of dim num_train_samples * num_labels.
        Both inputs are numpy arrays.
        For classification, labels could be either numbers 0, 1, ... c-1 for c classe
        or one-hot encoded vector of zeros, with a 1 at the kth position for class k.
        The AutoML format support on-hot encoding, which also works for multi-labels problems.
        Use data_converter.convert_to_num() to convert to the category number format.
        For regression, labels are continuous values.
        '''
        self.num_train_samples = X.shape[0]
        if X.ndim>1: self.num_feat = X.shape[1]
        print("FIT: dim(X)= [{:d}, {:d}]".format(self.num_train_samples, self.num_feat))
        num_train_samples = y.shape[0]
        if y.ndim>1: self.num_labels = y.shape[1]
        print("FIT: dim(y)= [{:d}, {:d}]".format(num_train_samples, self.num_labels))
        if (self.num_train_samples != num_train_samples):
            print("ARRGH: number of samples in X and y do not match!")

        '''For this baseline we will do a regular regression wihout taking
        censored data into account, the target value of the regression is contained in
        the first column of the y array'''
        # We get the target for the regression (y[:,1] contains the events)
        # y[:,0] is a slicing method, it takes all the lines ':' , and only the first column '0'
        # of the 2-d ndarray y
        y = y[:,0]
        # Once we have our regression target, we simply fit our model :
        self.baseline_clf.fit(X, y)
        self.is_trained=True
        
    def processor(self,X,Y, what):                 # Le preprocessing ( séparation des données censurés et non censurés )
    ###    NEW CONTRIBUTION OH THE GROUP ###       # Les tests sont sur le jupyter notebook
        '''This function should preprocess the data
        Args :
            X : training data matrix
            Y : Training label matrix
            what : if 0 return uncensored data, if 1 return censored data
        '''
        if (what==0) :                                              # Les données non censuré
            X_uncensored = X[np.where(X[:,X.shape[1]-1]==1)]
            if Y.shape[0]==0 :
                return (X_uncensored)
            else :
                Y_uncensored = Y[np.where(X[:,X.shape[1]-1]==1)]
                return (X_uncensored,Y_uncensored)
        if (what==1) :                                              # Les données censuré
            X_censored = X[np.where(X[:,X.shape[1]-1]==0)]
            if Y.shape[0]==0 :
                return (X_censored)
            else :
                Y_censored = Y[np.where(X[:,X.shape[1]-1]==1)]
                return (X_censored,Y_censored)
        

    def predict(self, X):
        '''
        This function should provide predictions of labels on (test) data.
        Here we just return zeros...
        Make sure that the predicted values are in the correct format for the scoring
        metric. For example, binary classification problems often expect predictions
        in the form of a discriminant value (if the area under the ROC curve it the metric)
        rather that predictions of the class labels themselves. For multi-class or multi-labels
        problems, class probabilities are often expected if the metric is cross-entropy.
        Scikit-learn also has a function predict-proba, we do not require it.
        The function predict eventually can return probabilities.
        '''
        num_test_samples = X.shape[0]
        if X.ndim>1: num_feat = X.shape[1]
        print("PREDICT: dim(X)= [{:d}, {:d}]".format(num_test_samples, num_feat))
        if (self.num_feat != num_feat):
            print("ARRGH: number of features in X does not match training data!")
        print("PREDICT: dim(y)= [{:d}, {:d}]".format(num_test_samples, self.num_labels))
        # We ask the model to predict new data X :
        pred = self.baseline_clf.predict(X)
        print('DEBUG : '+str(pred.shape))
        return pred

    def save(self, path="./"):
        pickle.dump(self, open(path + '_model.pickle', "wb"))

    def load(self, path="./"):
        modelfile = path + '_model.pickle'
        if isfile(modelfile):
            with open(modelfile, 'rb') as f:
                self = pickle.load(f)
            print("Model reloaded from: " + modelfile)
        return self

