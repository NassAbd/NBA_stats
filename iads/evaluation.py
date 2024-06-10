# -*- coding: utf-8 -*-

"""
Package: iads
File: evaluation.py
Année: LU3IN026 - semestre 2 - 2023-2024, Sorbonne Université
"""

# ---------------------------
# Fonctions d'évaluation de classifieurs

# import externe
import numpy as np
import pandas as pd
import copy

# ------------------------ 
#TODO: à compléter  plus tard
# ------------------------ 

def crossval(X, Y, n_iterations, iteration):
    Xtes = [i for i in range(int(iteration*len(X)/n_iterations), int((iteration+1)*len(X)/n_iterations))]
    Ytest = [Y[i] for i in range(int(iteration*len(X)/n_iterations), int((iteration+1)*len(X)/n_iterations))]
    Xapp = [X[i] for i in range(len(X)) if i not in Xtes]
    Yapp = [Y[i] for i in range(len(X)) if i not in Xtes]
    Xtest = [X[i] for i in Xtes.copy()]
    
    return np.array(Xapp), np.array(Yapp), np.array(Xtest), np.array(Ytest)
    
# ------------------------ 
"""
def crossval_strat(X, Y, n_iterations, iteration):
    Y_pos = []
    Y_neg = []
    for i in range(len(Y)):
        if Y[i] == 1:
            Y_pos.append(i)
        else:
            Y_neg.append(i)
            
    Xtest_pos = [Y_pos[i] for i in range(int(iteration*len(Y_pos)/n_iterations), int((iteration+1)*len(Y_pos)/n_iterations))]
    Xtest_neg = [Y_neg[i] for i in range(int(iteration*len(Y_neg)/n_iterations), int((iteration+1)*len(Y_neg)/n_iterations))]
    Xtes =  Xtest_neg + Xtest_pos 
    Ytest = [Y[i] for i in Xtes]
    Xapp = [X[i] for i in range(len(X)) if i not in Xtes]
    Yapp = [Y[i] for i in range(len(X)) if i not in Xtes]
    Xtest = [X[i] for i in Xtes.copy()]
    
   
    
    return np.array(Xapp), np.array(Yapp), np.array(Xtest), np.array(Ytest)
    """
def crossval_strat(X, Y, n_iterations, iteration):
    # Index des éléments positifs et négatifs
    Y_pos_idx = np.where(Y == 1)[0]
    Y_neg_idx = np.where(Y != 1)[0]

    # Sélection des indices pour les échantillons de test
    start_pos = int(iteration * len(Y_pos_idx) / n_iterations)
    end_pos = int((iteration + 1) * len(Y_pos_idx) / n_iterations)
    start_neg = int(iteration * len(Y_neg_idx) / n_iterations)
    end_neg = int((iteration + 1) * len(Y_neg_idx) / n_iterations)

    Xtest_pos_idx = Y_pos_idx[start_pos:end_pos]
    Xtest_neg_idx = Y_neg_idx[start_neg:end_neg]

    # Concaténation des indices des échantillons de test
    Xtes_idx = np.concatenate((Xtest_neg_idx, Xtest_pos_idx))

    # Extraction des échantillons de test et d'apprentissage
    Xtest = X[Xtes_idx]
    Ytest = Y[Xtes_idx]

    Xapp_idx = np.setdiff1d(np.arange(len(X)), Xtes_idx)
    Xapp = X[Xapp_idx]
    Yapp = Y[Xapp_idx]

    return Xapp, Yapp, Xtest, Ytest

# ------------------------ 

def analyse_perfs(L):
    """ L : liste de nombres réels non vide
        rend le tuple (moyenne, écart-type)
    """
    ecart= np.std(L)
    moyenne = np.mean(L)
    return (moyenne, ecart)
 
 
# ------------------------ 
   
def validation_croisee(C, DS, nb_iter):
    """ Classifieur * tuple[array, array] * int -> tuple[ list[float], float, float]
    """
    #############
    # A COMPLETER
    ############# 
    perf = []
    classifRef = C
    for i in range(nb_iter):
        currentClassif = copy.deepcopy(classifRef)
        Xapp,Yapp,Xtest,Ytest = crossval_strat(DS[0], DS[1], nb_iter, i)
        currentClassif.train(Xapp, Yapp)
        perf.append(currentClassif.accuracy(Xtest, Ytest))
        print("Iteration : ", i, "taille base app.=",len(Xapp),"taille base test=",len(Xtest),"Taux de bonne classif: ",currentClassif.accuracy(Xtest, Ytest))
    moyenne, ecartT = analyse_perfs(perf)
    res = (perf, moyenne, ecartT)
    return res

# ------------------------ 

def leave_one_out(C, DS):
    """ Classifieur * tuple[array, array] -> float
    """
    ###################### A COMPLETER 
    DS_desc = DS[0]
    DS_label = DS[1]
    nb_point = 0
    n = len(DS_desc)
    print(DS_desc)
    print()
    for i in range(len(DS_desc)):
        Arbreclass = copy.deepcopy(C)
        desc = copy.deepcopy(DS_desc)
        label = copy.deepcopy(DS_label)
        np.delete(desc, i,0)
        np.delete(label, i,0)
        print(desc)
        print()
        Arbreclass.train(desc ,label)
        if Arbreclass.predict(DS_desc[i]) == DS_label[i]:
            nb_point += 1
    return nb_point/n
            
        
        

    #################################
