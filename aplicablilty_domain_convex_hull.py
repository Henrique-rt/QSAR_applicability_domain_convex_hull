# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem.Fingerprints import FingerprintMols


from scipy.spatial import ConvexHull, convex_hull_plot_2d
from scipy.optimize import linprog
from sklearn import manifold

'''
text file with molecules in convex_hull
'''
def text_file(coords_training, coords_test,list_test_name,coords_external,list_external_name):
    file_external = open("molecules_of_external_set_in_AD.txt", "w")
    file_test= open("molecules_of_test_set_in_AD.txt", "w")
    n_points = len(coords_training)
    n_dim = 2 #to represent the scatter plot in a plane
    c = np.zeros(n_points)
    A = np.r_[coords_training.T,np.ones((1,n_points))]
    for i in range(len(coords_external)):
        b = np.r_[coords_external[i], np.ones(1)]
        lp = linprog(c, A_eq=A, b_eq=b)
        if lp.success is True:
            file_external.write("%s \n" %list_external_name[i])
    for i in range(len(coords_test)):
        b = np.r_[coords_test[i], np.ones(1)]
        lp = linprog(c, A_eq=A, b_eq=b)
        if lp.success is True:
            file_test.write("%s \n" %list_test_name[i])

'''
path to input structures
'''
path_training_set=" ~/Path/To/training_set" #training_set folder
path_test_set="~/Path/To/test_set"
path_external_set="~/Path/To/external_set" #set not used to build model


'''
import structures
'''
list_training_name=[] #training set
list_training_fp=[]
for filenames in os.listdir(path_training_set): 
    if filenames.endswith(".sdf"): #".mol2, .sdf"
        suppl = Chem.SDMolSupplier(path_training_set+filenames)
        for mol in suppl:
            fgp=FingerprintMols.FingerprintMol(mol)
            list_training_name.append(filenames)
            list_training_fp.append(fgp)

list_test_name=[] #test set
list_test_fp=[]
for filenames in os.listdir(path_test_set): 
    if filenames.endswith(".sdf"):
        suppl = Chem.SDMolSupplier(path_test_set+filenames)
        for mol in suppl:
            fgp=FingerprintMols.FingerprintMol(mol)
            list_test_name.append(filenames)
            list_test_fp.append(fgp)

list_external_name=[] #external set
list_external_fp=[]
for filenames in os.listdir(path_external_set): 
    if filenames.endswith(".sdf"):
        suppl = Chem.SDMolSupplier(path_external_set+filenames)
        for mol in suppl:
            fgp=FingerprintMols.FingerprintMol(mol)
            list_external_name.append(filenames)
            list_external_fp.append(fgp)

list_data_set=list_training_name+list_test_name+list_external_name #all data set-> training+test+external
list_data_set_fp=list_training_fp+list_test_fp+list_external_fp #all data set-> training+test+external


size=len(list_data_set_fp)
table=pd.DataFrame()
for m, i in enumerate(list_data_set_fp):
    for n, j in enumerate(list_data_set_fp):
        similarity=DataStructs.FingerprintSimilarity(i,j)
        table.loc[list_data_set[m],list_data_set[n]]=similarity

'''
Multidimensional scaling
'''
mds = manifold.MDS(n_components=2, dissimilarity="euclidean", random_state=6)
results = mds.fit(table)
coords = results.embedding_


coords= (coords - np.min(coords)) / (np.max(coords) - np.min(coords))#normalize

coords_training=coords[:len(list_training_fp)]
coords_test=coords[len(list_training_fp):len(list_test_fp)+len(list_training_fp)]
coords_external=coords[len(list_test_fp)+len(list_training_fp):]

text_file(coords_training, coords_test,list_test_name,coords_external,list_external_name)

plt.scatter(coords_training[:, 0], coords_training[:, 1], marker = 's',label='Training')#training
for label, x, y in zip(list_data_set[:len(list_training_fp)], coords_training[:, 0], coords_training[:, 1]): #show molecule name 
    plt.annotate(label,xy = (x, y), xytext = (0, 0),textcoords = 'offset pixels', ha = 'center', va = 'bottom', fontsize=8) #show molecule name 

'''
Visualize the convex hull
'''
hull = ConvexHull(coords_training)
for simplex in hull.simplices:
    plt.plot(coords_training[simplex, 0], coords_training[simplex, 1], 'k-')

plt.scatter(coords_test[:, 0], coords_test[:, 1], marker = '^',label='Test')#test
for label, x, y in zip(list_data_set[len(list_training_fp):len(list_test_fp)+len(list_training_fp)], coords_test[:, 0], coords_test[:, 1]): #show molecule name 
    plt.annotate(label,xy = (x, y), xytext = (0, 0),textcoords = 'offset pixels', ha = 'center', va = 'bottom', fontsize=8) #show molecule name 

plt.scatter(coords_external[:, 0], coords_external[:, 1], marker = 'X', label='External')#external
for label, x, y in zip(list_data_set[len(list_test_fp)+len(list_training_fp):], coords_external[:, 0], coords_external[:, 1]): #show molecule name 
    plt.annotate(label,xy = (x, y), xytext = (0, 0),textcoords = 'offset pixels', ha = 'center', va = 'bottom', fontsize=8)#show molecule name 
plt.legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",mode="expand", borderaxespad=0, ncol=3,shadow=True, fontsize='8')
plt.xlabel("MDS1",fontweight='bold')
plt.ylabel("MDS2",fontweight='bold')
plt.show()



