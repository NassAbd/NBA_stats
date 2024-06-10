# -*- coding: utf-8 -*-

"""
Package: iads
File: Clustering.py
Année: LU3IN026 - semestre 2 - 2023-2024, Sorbonne Université
"""

# ---------------------------
# Fonctions de Clustering

# import externe
import numpy as np
import pandas as pd
import scipy.cluster.hierarchy
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# ------------------------ 

def normalisation(df):
    # Copie du DataFrame pour éviter de modifier l'original
    df_norm = df.copy()
    
    # Normalisation de chaque colonne
    for colonne in df_norm.columns:
        # Trouver la valeur minimale et maximale de la colonne
        min_colonne = df_norm[colonne].min()
        max_colonne = df_norm[colonne].max()
        
        # Normalisation des valeurs de la colonne entre 0 et 1
        df_norm[colonne] = (df_norm[colonne] - min_colonne) / (max_colonne - min_colonne)
    
    return df_norm

# ------------------------ 

def dist_euclidienne(x, y) : 
    # Conversion des DataFrame pandas en tableaux NumPy
    if isinstance(x, pd.DataFrame):
        x = x.values
    if isinstance(y, pd.DataFrame):
        y = y.values
    
    # Calcul de la distance euclidienne entre les vecteurs x et y
    distance = np.linalg.norm(x - y)
    
    return distance

# ------------------------ 

def centroide(data):
    # Conversion des DataFrame pandas en tableaux NumPy
    if isinstance(data, pd.DataFrame):
        data = data.values
    
    # Calcul du centroïde
    centroide = np.mean(data, axis=0)
    
    return centroide

# ------------------------ 

def dist_centroides(groupe1, groupe2):
    # Calcul des centroïdes des deux groupes
    centroide_groupe1 = centroide(groupe1)
    centroide_groupe2 = centroide(groupe2)
    
    # Calcul de la distance euclidienne entre les centroïdes
    distance = dist_euclidienne(centroide_groupe1, centroide_groupe2)
    
    return distance

# ------------------------ 

def dist_average(dt1, dt2):
    if isinstance(dt1, pd.DataFrame):
        dt1 = dt1.values
    if isinstance(dt2, pd.DataFrame):
        dt2 = dt2.values
    somme = 0
    nb_elem = len(dt1)*len(dt2)
    for elm1 in dt1:
        for elm2 in dt2:
            somme += np.linalg.norm(elm1 - elm2)
    return somme/nb_elem 

# ------------------------  

def dist_complete(dt1, dt2):
    if isinstance(dt1, pd.DataFrame):
        dt1 = dt1.values
    if isinstance(dt2, pd.DataFrame):
        dt2 = dt2.values
    maximum = 0
    couple = ()
    for i in range(len(dt1)):
        for j in range(len(dt2)):
            dist = np.linalg.norm(dt1[i] - dt2[j])
            if dist > maximum:
                maximum = dist
                couple = (j, i)
    return maximum

# ------------------------  

def dist_simple(dt1, dt2):
    if isinstance(dt1, pd.DataFrame):
        dt1 = dt1.values
    if isinstance(dt2, pd.DataFrame):
        dt2 = dt2.values
    minimum = -1
    couple = ()
    for i in range(len(dt1)):
        for j in range(len(dt2)):
            dist = np.linalg.norm(dt1[i] - dt2[j])
            if dist < minimum or minimum ==-1:
                minimum = dist
                couple = (j, i)
    return minimum

# ------------------------  
    
def initialise_CHA(df):
    # Initialisation de la partition
    partition = {}
    
    # Parcourir chaque exemple dans le DataFrame
    for i, exemple in enumerate(df.values):
        partition[i] = [i]
    
    return partition

# ------------------------ 

def fusionne(df, partition, dist_fn=dist_centroides, verbose=False) :
    # distances entre paires de clusters
    mini = -1
    couple = ()
    valeur = ()
    cle_max = -1
    for key1, val1 in partition.items() : 
        if key1 > cle_max : 
            cle_max = key1
        for key2, val2 in partition.items() : 
            if key1 < key2 : 
                dist  = dist_fn(df.iloc[val1], df.iloc[val2])
                if mini == -1:
                    mini = dist
                    couple = (key1, key2)
                    valeur = (val1, val2)
                elif mini > dist:
                    mini = dist
                    couple = (key1, key2)
                    valeur = (val1, val2)
    n = cle_max + 1
    p1 = partition.copy()
    p1.pop(couple[0])
    p1.pop(couple[1])
    p1[n] = valeur[0]+valeur[1]
    if verbose :
        print("fusionne: distance mininimale trouvée entre", p1[n],  " = " , mini)
        print("fusionne: les 2 clusters dont les clés sont ", couple  ," sont fusionnés")
        print("fusionne: on crée la  nouvelle clé", n , "dans le dictionnaire.")
        print("fusionne: les clés de" , couple  ,"sont supprimées car leurs clusters ont été fusionnés.")
    return (p1, couple[0], couple[1],mini) 

# ------------------------ 

def CHA_centroid(df, verbose=False, dendrogramme=False) : 
    partition_init = initialise_CHA(df)
    liste_res = []
    taille = len(partition_init)
    while len(partition_init) > 1 : 
        (p1, couple0, couple1,mini) = fusionne(df, partition_init,dist_centroides, verbose)
        partition_init = p1
        elt = [couple0, couple1, mini,len(p1[taille]) ]
        taille +=1 
        liste_res.append(elt)
        if verbose:
            print()
    if dendrogramme:
        plt.figure(figsize=(30, 15)) # taille : largeur x hauteur
        plt.title('Dendrogramme', fontsize=25)    
        plt.xlabel("Indice d'exemple", fontsize=25)
        plt.ylabel('Distance', fontsize=25)
        # Construction du dendrogramme pour notre clustering :
        scipy.cluster.hierarchy.dendrogram(
            liste_res, 
            leaf_font_size=24.,  # taille des caractères de l'axe des X
        )
        
        # Affichage du résultat obtenu:
        plt.show()
    return liste_res


def CHA_centroid10(df, verbose=False, dendrogramme=False):
    partition_init = initialise_CHA(df)
    liste_res = []
    taille = len(partition_init)
    while len(partition_init) > 1:
        (p1, couple0, couple1, mini) = fusionne(df, partition_init, dist_centroides, verbose)
        partition_init = p1
        elt = [couple0, couple1, mini, len(p1[taille])]
        taille += 1
        liste_res.append(elt)
        if verbose:
            print()
    
    # Obtenir les 10 clusters
    # Construction de la matrice de liaison
    Z = np.array(liste_res)
    # Coupe de l'arbre pour obtenir 10 clusters
    groupes = scipy.cluster.hierarchy.fcluster(Z, 10, criterion='maxclust')
    # Création d'un dictionnaire pour stocker les groupes résultants
    resultats = {}
    for i, groupe in enumerate(groupes):
        if groupe not in resultats:
            resultats[groupe] = []
        resultats[groupe].append(i)
    # Affichage des groupes résultants
    print("Groupes résultants avec 10 clusters :")
    print(resultats)
    
    if dendrogramme:
        plt.figure(figsize=(30, 15)) # taille : largeur x hauteur
        plt.title('Dendrogramme', fontsize=25)    
        plt.xlabel("Indice d'exemple", fontsize=25)
        plt.ylabel('Distance', fontsize=25)
        # Construction du dendrogramme pour notre clustering :
        scipy.cluster.hierarchy.dendrogram(
            liste_res, 
            leaf_font_size=24.,  # taille des caractères de l'axe des X
        )
        
        # Affichage du résultat obtenu:
        plt.show()
    
    return liste_res, resultats

    
# ------------------------ 

def CHA_simple(df, verbose=False, dendrogramme=False) : 
    partition_init = initialise_CHA(df)
    liste_res = []
    taille = len(partition_init)
    while len(partition_init) > 1 : 
        (p1, couple0, couple1,mini) = fusionne(df, partition_init,dist_simple, verbose)
        partition_init = p1
        elt = [couple0, couple1, mini,len(p1[taille]) ]
        taille +=1 
        liste_res.append(elt)
        if verbose:
            print()
    if dendrogramme:
        plt.figure(figsize=(30, 15)) # taille : largeur x hauteur
        plt.title('Dendrogramme', fontsize=25)    
        plt.xlabel("Indice d'exemple", fontsize=25)
        plt.ylabel('Distance', fontsize=25)
        # Construction du dendrogramme pour notre clustering :
        scipy.cluster.hierarchy.dendrogram(
            liste_res, 
            leaf_font_size=24.,  # taille des caractères de l'axe des X
        )
        
        # Affichage du résultat obtenu:
        plt.show()
    return liste_res


def CHA_simple10(df, verbose=False, dendrogramme=False):
    partition_init = initialise_CHA(df)
    liste_res = []
    taille = len(partition_init)
    while len(partition_init) > 1:
        (p1, couple0, couple1, mini) = fusionne(df, partition_init, dist_simple, verbose)
        partition_init = p1
        elt = [couple0, couple1, mini, len(p1[taille])]
        taille += 1
        liste_res.append(elt)
        if verbose:
            print()
    
    # Obtenir les 10 clusters
    # Construction de la matrice de liaison
    Z = np.array(liste_res)
    # Coupe de l'arbre pour obtenir 10 clusters
    groupes = scipy.cluster.hierarchy.fcluster(Z, 10, criterion='maxclust')
    # Création d'un dictionnaire pour stocker les groupes résultants
    resultats = {}
    for i, groupe in enumerate(groupes):
        if groupe not in resultats:
            resultats[groupe] = []
        resultats[groupe].append(i)
    # Affichage des groupes résultants
    print("Groupes résultants avec 10 clusters :")
    print(resultats)
    
    if dendrogramme:
        plt.figure(figsize=(30, 15)) # taille : largeur x hauteur
        plt.title('Dendrogramme', fontsize=25)    
        plt.xlabel("Indice d'exemple", fontsize=25)
        plt.ylabel('Distance', fontsize=25)
        # Construction du dendrogramme pour notre clustering :
        scipy.cluster.hierarchy.dendrogram(
            liste_res, 
            leaf_font_size=24.,  # taille des caractères de l'axe des X
        )
        
        # Affichage du résultat obtenu:
        plt.show()
    
    return liste_res, resultats

# ------------------------ 

def CHA_average(df, verbose=False, dendrogramme=False) : 
    partition_init = initialise_CHA(df)
    liste_res = []
    taille = len(partition_init)
    while len(partition_init) > 1 : 
        (p1, couple0, couple1,mini) = fusionne(df, partition_init,dist_average, verbose)
        partition_init = p1
        elt = [couple0, couple1, mini,len(p1[taille]) ]
        taille +=1 
        liste_res.append(elt)
        if verbose:
            print()
    if dendrogramme:
        plt.figure(figsize=(30, 15)) # taille : largeur x hauteur
        plt.title('Dendrogramme', fontsize=25)    
        plt.xlabel("Indice d'exemple", fontsize=25)
        plt.ylabel('Distance', fontsize=25)
        # Construction du dendrogramme pour notre clustering :
        scipy.cluster.hierarchy.dendrogram(
            liste_res, 
            leaf_font_size=24.,  # taille des caractères de l'axe des X
        )
        
        # Affichage du résultat obtenu:
        plt.show()
    return liste_res


def CHA_average10(df, verbose=False, dendrogramme=False):
    partition_init = initialise_CHA(df)
    liste_res = []
    taille = len(partition_init)
    while len(partition_init) > 1:
        (p1, couple0, couple1, mini) = fusionne(df, partition_init, dist_average, verbose)
        partition_init = p1
        elt = [couple0, couple1, mini, len(p1[taille])]
        taille += 1
        liste_res.append(elt)
        if verbose:
            print()
    
    # Obtenir les 10 clusters
    # Construction de la matrice de liaison
    Z = np.array(liste_res)
    # Coupe de l'arbre pour obtenir 10 clusters
    groupes = scipy.cluster.hierarchy.fcluster(Z, 10, criterion='maxclust')
    # Création d'un dictionnaire pour stocker les groupes résultants
    resultats = {}
    for i, groupe in enumerate(groupes):
        if groupe not in resultats:
            resultats[groupe] = []
        resultats[groupe].append(i)
    # Affichage des groupes résultants
    print("Groupes résultants avec 10 clusters :")
    print(resultats)
    
    if dendrogramme:
        plt.figure(figsize=(30, 15)) # taille : largeur x hauteur
        plt.title('Dendrogramme', fontsize=25)    
        plt.xlabel("Indice d'exemple", fontsize=25)
        plt.ylabel('Distance', fontsize=25)
        # Construction du dendrogramme pour notre clustering :
        scipy.cluster.hierarchy.dendrogram(
            liste_res, 
            leaf_font_size=24.,  # taille des caractères de l'axe des X
        )
        
        # Affichage du résultat obtenu:
        plt.show()
    
    return liste_res, resultats

# ------------------------ 

def CHA_complete(df, verbose=False, dendrogramme=False) : 
    partition_init = initialise_CHA(df)
    liste_res = []
    taille = len(partition_init)
    while len(partition_init) > 1 : 
        (p1, couple0, couple1,mini) = fusionne(df, partition_init,dist_complete, verbose)
        partition_init = p1
        elt = [couple0, couple1, mini,len(p1[taille]) ]
        taille +=1 
        liste_res.append(elt)
        if verbose:
            print()
    if dendrogramme:
        plt.figure(figsize=(30, 15)) # taille : largeur x hauteur
        plt.title('Dendrogramme', fontsize=25)    
        plt.xlabel("Indice d'exemple", fontsize=25)
        plt.ylabel('Distance', fontsize=25)
        # Construction du dendrogramme pour notre clustering :
        scipy.cluster.hierarchy.dendrogram(
            liste_res, 
            leaf_font_size=24.,  # taille des caractères de l'axe des X
        )
        
        # Affichage du résultat obtenu:
        plt.show()
    return liste_res


def CHA_complete10(df, verbose=False, dendrogramme=False):
    partition_init = initialise_CHA(df)
    liste_res = []
    taille = len(partition_init)
    while len(partition_init) > 1:
        (p1, couple0, couple1, mini) = fusionne(df, partition_init, dist_complete, verbose)
        partition_init = p1
        elt = [couple0, couple1, mini, len(p1[taille])]
        taille += 1
        liste_res.append(elt)
        if verbose:
            print()
    
    # Obtenir les 10 clusters
    # Construction de la matrice de liaison
    Z = np.array(liste_res)
    # Coupe de l'arbre pour obtenir 10 clusters
    groupes = scipy.cluster.hierarchy.fcluster(Z, 10, criterion='maxclust')
    # Création d'un dictionnaire pour stocker les groupes résultants
    resultats = {}
    for i, groupe in enumerate(groupes):
        if groupe not in resultats:
            resultats[groupe] = []
        resultats[groupe].append(i)
    # Affichage des groupes résultants
    print("Groupes résultants avec 10 clusters :")
    print(resultats)
    
    if dendrogramme:
        plt.figure(figsize=(30, 15)) # taille : largeur x hauteur
        plt.title('Dendrogramme', fontsize=25)    
        plt.xlabel("Indice d'exemple", fontsize=25)
        plt.ylabel('Distance', fontsize=25)
        # Construction du dendrogramme pour notre clustering :
        scipy.cluster.hierarchy.dendrogram(
            liste_res, 
            leaf_font_size=24.,  # taille des caractères de l'axe des X
        )
        
        # Affichage du résultat obtenu:
        plt.show()
    
    return liste_res, resultats

# ------------------------ 

def CHA(DF,linkage='centroid', verbose=False,dendrogramme=False):
    """  ##### donner une documentation à cette fonction
    """
    ############################ A COMPLETER
    if linkage == 'centroid' :
        return CHA_centroid(DF, verbose, dendrogramme) 
    elif linkage == 'complete':
        return CHA_complete(DF, verbose, dendrogramme) 
    elif linkage == 'simple':
        return CHA_simple(DF, verbose, dendrogramme) 
    elif linkage == 'average':
        return CHA_average(DF, verbose, dendrogramme) 

    raise NotImplementedError("Please Implement this method")
    
# ------------------------ 
        
def init_kmeans(K,Ens):
    """ int * Array -> Array
        K : entier >1 et <=n (le nombre d'exemples de Ens)
        Ens: Array contenant n exemples
    """

    ############# A COMPLETER 
    if isinstance(Ens, pd.DataFrame):
        Ens = Ens.values
    
    # Sélectionner aléatoirement k indices
    indices_aleatoires = np.random.choice(len(Ens), K, replace=False)
    
    # Sélectionner les exemples correspondants à partir des indices
    exemples_aleatoires = Ens[indices_aleatoires]
    
    return exemples_aleatoires

# ------------------------ 

def affecte_cluster(Base,Centres):
    """ Array * Array -> dict[int,list[int]]
        Base: Array contenant la base d'apprentissage
        Centres : Array contenant des centroides
    """
    
    ############# A COMPLETER 
    if isinstance(Base, pd.DataFrame):
        Base = Base.values
    
    matrice_affectation = {}
    for i, exemple in enumerate(Base): #pour itérer sur Base et avoir son indice dans i
        indice_cluster = plus_proche(exemple, Centres)
        if indice_cluster not in matrice_affectation:
            matrice_affectation[indice_cluster] = [i]
        else:
            matrice_affectation[indice_cluster].append(i)
    return matrice_affectation   
 
# ------------------------ 

def plus_proche(Exe,Centres):
    """ Array * Array -> int
        Exe : Array contenant un exemple
        Centres : Array contenant les K centres
    """

    ############# A COMPLETER 
    distances = [dist_euclidienne(Exe, centroid) for centroid in Centres]
    
    return np.argmin(distances)

# ------------------------ 

def nouveaux_centroides(Base,U):
    """ Array * dict[int,list[int]] -> DataFrame
        Base : Array contenant la base d'apprentissage
        U : Dictionnaire d'affectation
    """
    
    ############# A COMPLETER 
    if isinstance(Base, pd.DataFrame):
        Base = Base.values

    nouveaux_centroides = []
    for cluster_indices in U.values():
        exemples_cluster = Base[cluster_indices]
        nouveau_centroide = np.mean(exemples_cluster, axis=0)
        nouveaux_centroides.append(nouveau_centroide)
    return np.array(nouveaux_centroides)

# ------------------------ 

def inertie_cluster(Ens):
    """ Array -> float
        Ens: array qui représente un cluster
        Hypothèse: len(Ens)> >= 2
        L'inertie est la somme (au carré) des distances des points au centroide.
    """

    ############# A COMPLETER 
    # Calculer le centre de l'ensemble (moyenne)
    centroide = np.mean(Ens, axis=0)
    
    # Calculer les distances de chaque point par rapport au centre
    distances = np.linalg.norm(Ens - centroide, axis=1)
    
    # Calculer l'inertie en prenant la somme des carrés des distances
    inertie = np.sum(distances ** 2)

    return inertie
    
# ------------------------ 

def inertie_globale(Base, U):
    """ Array * dict[int,list[int]] -> float
        Base : Array pour la base d'apprentissage
        U : Dictionnaire d'affectation
    """
    
    ############# A COMPLETER 
    if isinstance(Base, pd.DataFrame):
        Base = Base.values
    
    inertie_globale = 0.0
    for cluster_indices in U.values():
        exemples_cluster = Base[cluster_indices]
        inertie_globale += inertie_cluster(exemples_cluster)
    return inertie_globale

# ------------------------ 
    
def kmoyennes(K, Base, epsilon, iter_max):
    """ int * Array * float * int -> tuple(Array, dict[int,list[int]])
        K : entier > 1 (nombre de clusters)
        Base : Array pour la base d'apprentissage
        epsilon : réel >0
        iter_max : entier >1
    """
    
    ############# A COMPLETER

    # Initialisation des centroides
    centroides = init_kmeans(K, Base)
    U = None
    for i in range(iter_max):
        # Affectation des exemples aux clusters
        U = affecte_cluster(Base, centroides)
        # Mise à jour des centroides
        nouveaux_centres = nouveaux_centroides(Base, U)
        # Calcul de la différence entre les centroides actuels et précédents
        diff = np.linalg.norm(nouveaux_centres - centroides)

        inertie = inertie_globale(Base, U)
        print("iteration ", i+1, " Inertie : ", inertie, " Difference : ", diff)
        
        # Vérification de la convergence
        if diff < epsilon:
            break
        centroides = nouveaux_centres
    return centroides, U



# ------------------------

def affiche_resultat(Base,Centres,Affect):
    """ DataFrame **2 * dict[int,list[int]] -> None
    """

    ############# A COMPLETER 
    if isinstance(Base, pd.DataFrame):
        Base = Base.values
    
    if len(Centres) > 20:
        print("Nombre de clusters supérieur à 20.")
        return

    # Création d'une colormap avec 20 couleurs différentes
    colors = cm.tab20(np.linspace(0, 1, len(Centres)))

    # Affichage des exemples de chaque cluster avec une couleur différente
    for cluster, exemples in Affect.items():
        couleur = colors[cluster]
        plt.scatter(Base[exemples, 0], Base[exemples, 1], color=couleur, label=f"Cluster {cluster}")

    # Affichage des centroides
    plt.scatter(Centres[:, 0], Centres[:, 1], marker='x', color='black', label='Centroides')

    plt.title('Clusters trouvés')
    plt.legend()
    plt.show()
    
    
    