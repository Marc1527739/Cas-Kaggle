# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 03:59:58 2022

@author: Marc
"""

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
import train_model
import generate_features
#En este script se podran realizar nuevas prediccions de un modelo ya entrenado en el script train_model
  
    
#Main
def main():
    dataset,dataset_N = generate_features.main()
    rlogistica,rl,svm,k_fold,randomforest,knn = train_model.main()
    
    print("Exemple de una nova predicció a partir d'un model ja creat':")
    print("Imaginem que volem fer una nova predicció amb el model de regressio logistica ja entrenat. ")
    print("El que hauriem de fer es preparar les dades de la nova prediccio. En aquest cas per exemple, cambiarem el conjunt d'aprenentage test-train...")
    Y = dataset.iloc[:,0].values
    X = dataset_N.iloc[:,1:39].values

    #Preparem el model
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.40, random_state=101)
    print("I realitzariem la prediccio amb el model ja entrenat que es troba declarat com ha objecte rlogistica")
    
    rlogistica.fit(X_train,y_train)
    y_pred = rlogistica.predict(X_test)

    print("Resultat de la nova predicció utilitzant un model ja entrenat:")
    #Evaluem els resultat
    print("Precisió: ",rlogistica.score(X_test,y_test))
    print("MSE: ",mse(y_pred,y_test))
    print("\n",confusion_matrix(y_test,y_pred))
    print("\n",classification_report(y_test,y_pred))
    
    
    
    

if __name__ == '__main__':
    main()