# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 03:25:11 2022

@author: Marc
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 02:48:47 2022

@author: Marc
"""
#LLibreries a importar

from sklearn.datasets import make_regression
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import scipy.stats
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold

#Métriques
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import classification_report,confusion_matrix
import time
from statsmodels.tools.eval_measures import mse

#Identificació hiperparametres
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
import generate_features

#En este script solo se añadira el entrenamiento de cada modelo que de mejores resultados
#Si se quieren ver otro tipo de entrenamientos, se encuentran en el notebook dentro del github

def RegressioLogistica(dataset,dataset_N):
    #Prenem les varibles dependents i la objectiu
    Y = dataset.iloc[:,0].values
    X = dataset_N.iloc[:,1:39].values

    #Preparem el model
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.60, random_state=101)
    model = LogisticRegression()

    #Entrenament
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)

    #Evaluem els resultat
    print("Precisió: ",model.score(X_test,y_test))
    print("MSE: ",mse(y_pred,y_test))
    print("\n",confusion_matrix(y_test,y_pred))
    print("\n",classification_report(y_test,y_pred))

    return model

def RegressioLineal(dataset,dataset_N):
    #Prenem les varibles dependents i la objectiu
    Y = dataset.iloc[:,0].values
    X = dataset_N.iloc[:,1:39].values

    #Preparem el model
    X_train, X_test, y_train, y_test = train_test_split(X, Y,test_size=0.60, random_state=101)
    model = LinearRegression()

    #Realitzem entrenament
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)

    #Evaluem els resultat
    print("Precisió: ",model.score(X_test,y_test))
    print("MSE: ",mse(y_pred,y_test))
    
    return model

def SVM_rbf(dataset):
    #Prenem les varibles dependents i la objectiu
    Y = dataset.iloc[:,0].values
    X = dataset.iloc[:,1:39].values

    #Preparem el model
    X_train, X_test, y_train, y_test = train_test_split(X, Y,test_size=0.60, random_state=101)
    svc = SVC(kernel='rbf')

    #Realitzem entrenament
    svc.fit(X_train,y_train)
    y_pred = svc.predict(X_test)

    #Evaluem els resultat
    print("Precisió: ",svc.score(X_test,y_test))
    print("MSE: ",mse(y_pred,y_test))
    print("\n",confusion_matrix(y_test,y_pred))
    print("\n",classification_report(y_test,y_pred))
    return svc

def K_fold(dataset,dataset_N):
    #Prenem les varibles dependents i la objectiu
    Y = dataset.iloc[:,0].values
    X = dataset_N.iloc[:,1:39].values

    X_train, X_test, y_train, y_test = train_test_split(X, Y,test_size=0.60,random_state = 101)

    #Escollim paràmetre k
    kf = KFold(n_splits=5,shuffle=True)
    for train_index, test_index in kf.split(X):
    
        #Preparem el model
        X_train, X_test, y_train, y_test = X[train_index],X[test_index],Y[train_index],Y[test_index]
                                    
    
        model = LogisticRegression()

        #Realitzem la predicció
        model.fit(X_train,y_train)
        y_pred = model.predict(X_test)
    
        #Evaluem els resultat
        print("Precisió: ",model.score(X_test,y_test))
        print("MSE: ",mse(y_pred,y_test))
        print("\n",confusion_matrix(y_test,y_pred))
        print("\n",classification_report(y_test,y_pred))

    return kf

def RandomForest(dataset,dataset_N):
    Y = dataset.iloc[:,0].values
    X = dataset_N.iloc[:,1:39].values

    #Preparem el model
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.60, random_state=101)
    model = RandomForestClassifier()

    #Realitzem entrenament
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)

    #Evaluem els resultat
    print("Precisió: ",model.score(X_test,y_test))
    print("MSE: ",mse(y_pred,y_test))
    print("\n",confusion_matrix(y_test,y_pred))
    print("\n",classification_report(y_test,y_pred))
    
    return model

def KNN(dataset,dataset_N):
    #Prenem les varibles dependents i la objectiu
    Y = dataset.iloc[:,0].values
    X = dataset_N.iloc[:,1:39].values

    #Preparem el model
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.60, random_state=101)
    model = KNeighborsClassifier()

    #Realitzem entrenament
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)

    #Evaluem els resultat
    print("Precisió: ",model.score(X_test,y_test))
    print("MSE: ",mse(y_pred,y_test))
    print("\n",confusion_matrix(y_test,y_pred))
    print("\n",classification_report(y_test,y_pred))
    
    return model

    
    
#Main
def main():
    #Entrenament de tots els models
    print("Els models s'entrenaran tots amb la base de dades normalitzada ja que es la que millors resultats dona")
    print("Excepte el model SVM que s'entrenara amb la base de dades original")
    dataset,dataset_N = generate_features.main()

    print("Entrenament model RegressioLogisitca...")
    rlogistica = RegressioLogistica(dataset, dataset_N)

    print("Entrenament model RegressioLineal...")
    rl = RegressioLineal(dataset, dataset_N)
    
    print("Entrenament model SVM amb kernel gaussiana...")
    svm = SVM_rbf(dataset)
    
    print("Entrenament model K_fold amb k = 5...")
    k_fold = K_fold(dataset, dataset_N)
    
    print("Entrenament model RandomForest...")
    randomforest = RandomForest(dataset, dataset_N)
    
    print("Entrenament model KNN...")
    knn = KNN(dataset, dataset_N)
    
    
    return rlogistica,rl,svm,k_fold,randomforest,knn

if __name__ == '__main__':
    main()