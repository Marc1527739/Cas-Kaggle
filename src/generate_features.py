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

#Limpieza dataset
def importar(path):
    dataset = pd.read_csv(path, header=0, delimiter=',', encoding = "ISO-8859-1")
    return dataset

#Transformacion_datos

    
def convertir_atributs_float(dataset):
   for col in dataset:
      dataset[col] = dataset[col].astype(float)
   return dataset   
            
def eliminar_columnes(dataset):
    dataset = dataset.drop(['gameId'], axis=1)
    return dataset

def normalitzacio(dataset):
    def standarize(x_train):
        mean = x_train.mean(0)
        std = x_train.std(0)
        x_t = x_train - mean[None, :]
        x_t /= std[None, :]
        return x_t
    dataset_N = dataset.copy()
    x = dataset_N
    dataset_N = standarize(dataset)
    return dataset_N
    
def PCA_4_components(dataset_N):
    datasetPCA1 = dataset_N.copy()
    nComponents = 4
    x1 = datasetPCA1.iloc[:,0:40].values
    pca = PCA(n_components = nComponents, svd_solver = 'full')
    principalComponents = pca.fit_transform(x1)
    principalDf = pd.DataFrame(data = principalComponents)
    datasetPCA = pd.concat([ principalDf], axis = 1)
    return datasetPCA
    
def TSNE_2_components(dataset_N):
    datasetTSNE = dataset_N.copy()
    nComponents = 2

    x1 = datasetTSNE.iloc[:,0:40].values

    tsne = TSNE(n_components = nComponents, random_state=123)
    principalComponents = tsne.fit_transform(x1)
    principalDf = pd.DataFrame(data = principalComponents)

    datasetTSNE = pd.concat([ principalDf], axis = 1)
    return datasetTSNE


#Main
def main():
    #Neteja
    dataset = importar('high_diamond_ranked_10min.csv')
    #Transformacio 
    dataset = convertir_atributs_float(dataset)
    dataset = eliminar_columnes(dataset)
    dataset_N = normalitzacio(dataset)
    dataset_PCA = PCA_4_components(dataset_N)
    dataset_TSNE = TSNE_2_components(dataset_N)
    
    #Resum dels quatre dataset nous
    print("Nou dataset original ")
    print("Dimensionalitat de la BBDD:", dataset.shape)
    print("\nTabla de la BBDD:")
    print(dataset.head())
    
    print("Nou dataset noramlitzat ")
    print("Dimensionalitat de la BBDD:", dataset_N.shape)
    print("\nTabla de la BBDD:")
    print(dataset_N.head())
    
    print("Nou dataset PCA ")
    print("Dimensionalitat de la BBDD:", dataset_PCA.shape)
    print("\nTabla de la BBDD:")
    print(dataset_PCA.head())
    
    print("Nou dataset TSNE ")
    print("Dimensionalitat de la BBDD:", dataset_TSNE.shape)
    print("\nTabla de la BBDD:")
    print(dataset_TSNE.head())
    
    return dataset,dataset_N
    
    
if __name__ == '__main__':
    main()