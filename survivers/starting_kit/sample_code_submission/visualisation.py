import numpy as np
import model as m
import mpca as mspca
from libscores import get_metric
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score
metric_name, scoring_function = get_metric()
from data_manager import DataManager
import matplotlib.pyplot as plt

##la plupart des graphes ont été réalisé sur excel à partir des résultats codalab le reste est dans ce fichier##
class visualisation :
    
    data_dir = '../public_data'              
    data_name = 'Mortality'
    D = DataManager(data_name, data_dir, replace_missing=True)
    
    X_train = D.data['X_train']
    Y_train = D.data['Y_train']
    
    
    '''cree une np.array avec les scores par cross- validation des differentes regression avec PCA 
    en fonction de n_compenent ''' # mais on obtient le meme scores pour chaque ligne 
    def array_pca :
        compteur = 0
        graph_array = np.empty((X_train.shape[1]* 6,3),dtype = np.object)
        for i in range (1, X_train.shape[1] +1) :
            for j in range (6) :
                M = m.model(j, i, True)
                M.fit(X_train , Y_train) 
                scores = cross_val_score(M, X_train, Y_train, cv=5, scoring=make_scorer(scoring_function))
                graph_array[compteur,0] = i #n_component
                graph_array[compteur,1] = M.baseline_clf.steps[1][0] #regression testé
                graph_array[compteur,2] = scores.mean() #score de cross-validation
                compteur += 1
    
    def array_sanspca :
        '''cree une np.array avec les scores par cross- validation des differentes regression sans PCA '''
        graph_array_sans_PCA = np.empty((6,2),dtype = np.object)
        nom_reg = ['DecisionTreeRegressor', 'GaussianNB', 'LinearRegression', 'RandomForestClassifier', 
                   ' NearestCentroid', 'GradientBoostingRegressor']
        compteur1 = 0
        for j in range (6) :
            M = mspca.model(j)
            M.fit(X_train , Y_train)
            scores = cross_val_score(M, X_train, Y_train, cv=5, scoring=make_scorer(scoring_function))
            graph_array_sans_PCA [compteur1,0] = nom_reg[compteur1]  #regression testé
            graph_array_sans_PCA [compteur1,1] = scores.mean()  #score de cross-validation
            compteur1 += 1
    if _name_ =='_main_ :
        '''creation du graph a partir de graph_array_sans_PCA'''
        fig,ax= plt.subplots(1,1,figsize=(15,5))
        graph_array_sans_PCA = array_sanspca()
        X= graph_array_sans_PCA[:,0]
        Y= graph_array_sans_PCA[:,1]
        ax.plot(X,Y,'^',)
        fig.suptitle("scores par cross-validation sans le PCA")
        
        plt.show()
