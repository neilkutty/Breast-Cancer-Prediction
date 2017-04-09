#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 19:02:44 2017

@author: NNK

Features are computed from a digitized image of a fine needle aspirate (FNA) 
of a breast mass. They describe characteristics of the cell nuclei present 
in the image. n the 3-dimensional space is that described in: 
[K. P. Bennett and O. L. Mangasarian: "Robust Linear Programming Discrimination of 
Two Linearly Inseparable Sets", Optimization Methods and Software 1, 1992, 23-34].

This database is also available through the UW CS ftp server: 
    ftp ftp.cs.wisc.edu cd math-prog/cpo-dataset/machine-learn/WDBC/

Also can be found on UCI Machine Learning Repository: 
    https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29

Attribute Information:

1) ID number 2) Diagnosis (M = malignant, B = benign) 

3-32)

Ten real-valued features are computed for each cell nucleus:

a) radius (mean of distances from center to points on the perimeter) 
b) texture (standard deviation of gray-scale values) 
c) perimeter 
d) area 
e) smoothness (local variation in radius lengths) 
f) compactness (perimeter^2 / area - 1.0) 
g) concavity (severity of concave portions of the contour) 
h) concave points (number of concave portions of the contour) 
i) symmetry 
j) fractal dimension ("coastline approximation" - 1)

The mean, standard error and "worst" or largest (mean of the three largest values)
of these features were computed for each image, resulting in 30 features.
For instance, field 3 is Mean Radius, field 13 is Radius SE, field 23 is Worst Radius.

All feature values are recoded with four significant digits.


"""
import numpy as np
import pandas as pd
import seaborn as s
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


#%% Get and Clean Data

d = pd.read_csv('diagnostic.csv')

df = d.drop('Unnamed: 32', axis=1)

#if using diagnosis as categorical
df.diagnosis = df.diagnosis.astype('category')

#Create references to subset predictor and outcome variables
x = list(df.drop('diagnosis',axis=1).drop('id',axis=1))
y ='diagnosis'

# -- Feature Normalization / Scaling -----------------------------------------
#  Normalize features for SVM and MLPClassifier
#-----------------------------------------------------------------------------
df2 = df[x]
df_norm = (df2 - df2.mean()) / (df2.max() - df2.min())
df_norm = pd.concat([df_norm, df[y]], axis=1)
#-----------------------------------------------------------------------------


#show first 10 rows
df.head(10)
#-----------------------------------------------------------------------------


#%% ##-----------------------------------------------------------##
##                  Exploratory                              ##
##-----------------------------------------------------------##
##-----------------------------------------------------------##
##-----------------------------------------------------------##

#%%
# Visualize Correlation among subsets of Predictors

mean_cols = [col for col in df.columns if 'mean' in col]
s.set(font_scale=1.4)
s.heatmap(df[mean_cols].corr(), cmap='coolwarm')


#%%

#Visualize Correlations among Predictors


s.set(font_scale=1.4)
s.heatmap(df.drop('diagnosis', axis=1).drop('id',axis=1).corr(), cmap='coolwarm')




#%% Explore 1

g = s.FacetGrid(df, col='diagnosis', hue='diagnosis')
g.map(s.distplot, "radius_mean", hist=False, rug=True)

g = s.FacetGrid(df, col='diagnosis', hue='diagnosis')
g.map(s.distplot, "texture_mean", hist=True, rug=True)

g = s.FacetGrid(df, col='diagnosis', hue='diagnosis')
g.map(s.distplot, "perimeter_mean", hist=True, rug=True)

g = s.FacetGrid(df, col='diagnosis', hue='diagnosis')
g.map(s.distplot, "area_mean", hist=True, rug=True)

g = s.FacetGrid(df, col='diagnosis', hue='diagnosis')
g.map(s.distplot, "smoothness_mean", hist=True, rug=True)

g = s.FacetGrid(df, col='diagnosis', hue='diagnosis')
g.map(s.distplot, "compactness_mean", hist=True, rug=True)

g = s.FacetGrid(df, col='diagnosis', hue='diagnosis')
g.map(s.distplot, "concavity_mean", hist=True, rug=True)

g = s.FacetGrid(df, col='diagnosis', hue='diagnosis')
g.map(s.distplot, "concave points_mean", hist=True, rug=True)

g = s.FacetGrid(df, col='diagnosis', hue='diagnosis')
g.map(s.distplot, "symmetry_mean", hist=True, rug=True)

g = s.FacetGrid(df, col='diagnosis', hue='diagnosis')
g.map(s.distplot, "fractal_dimension_mean", hist=True, rug=True)



#%% Explore 1-a.)
plt.rcParams['figure.figsize']=(10,5)
f, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1,5)
s.boxplot('diagnosis',y='radius_mean',data=df, ax=ax1)
s.boxplot('diagnosis',y='texture_mean',data=df, ax=ax2)
s.boxplot('diagnosis',y='perimeter_mean',data=df, ax=ax3)
s.boxplot('diagnosis',y='area_mean',data=df, ax=ax4)
s.boxplot('diagnosis',y='smoothness_mean',data=df, ax=ax5)
f.tight_layout()

f, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1,5)
s.boxplot('diagnosis',y='compactness_mean',data=df, ax=ax2)
s.boxplot('diagnosis',y='concavity_mean',data=df, ax=ax1)
s.boxplot('diagnosis',y='concave points_mean',data=df, ax=ax3)
s.boxplot('diagnosis',y='symmetry_mean',data=df, ax=ax4)
s.boxplot('diagnosis',y='fractal_dimension_mean',data=df, ax=ax5)    
f.tight_layout()

#%%

plt.rcParams['figure.figsize']=(10,5)

f, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1,5)
s.boxplot('diagnosis',y='radius_se',data=df, ax=ax1, palette='cubehelix')
s.boxplot('diagnosis',y='texture_se',data=df, ax=ax2, palette='cubehelix')
s.boxplot('diagnosis',y='perimeter_se',data=df, ax=ax3, palette='cubehelix')
s.boxplot('diagnosis',y='area_se',data=df, ax=ax4, palette='cubehelix')
s.boxplot('diagnosis',y='smoothness_se',data=df, ax=ax5, palette='cubehelix')
f.tight_layout()


f, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1,5)
s.boxplot('diagnosis',y='compactness_se',data=df, ax=ax2, palette='cubehelix')
s.boxplot('diagnosis',y='concavity_se',data=df, ax=ax1, palette='cubehelix')
s.boxplot('diagnosis',y='concave points_se',data=df, ax=ax3, palette='cubehelix')
s.boxplot('diagnosis',y='symmetry_se',data=df, ax=ax4, palette='cubehelix')
s.boxplot('diagnosis',y='fractal_dimension_se',data=df, ax=ax5, palette='cubehelix')    
f.tight_layout()

#%%
plt.rcParams['figure.figsize']=(10,5)
f, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1,5)
s.boxplot('diagnosis',y='radius_worst',data=df, ax=ax1, palette='coolwarm')
s.boxplot('diagnosis',y='texture_worst',data=df, ax=ax2, palette='coolwarm')
s.boxplot('diagnosis',y='perimeter_worst',data=df, ax=ax3, palette='coolwarm')
s.boxplot('diagnosis',y='area_worst',data=df, ax=ax4, palette='coolwarm')
s.boxplot('diagnosis',y='smoothness_worst',data=df, ax=ax5, palette='coolwarm')
f.tight_layout()

f, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1,5)
s.boxplot('diagnosis',y='compactness_worst',data=df, ax=ax2, palette='coolwarm')
s.boxplot('diagnosis',y='concavity_worst',data=df, ax=ax1, palette='coolwarm')
s.boxplot('diagnosis',y='concave points_worst',data=df, ax=ax3, palette='coolwarm')
s.boxplot('diagnosis',y='symmetry_worst',data=df, ax=ax4, palette='coolwarm')
s.boxplot('diagnosis',y='fractal_dimension_worst',data=df, ax=ax5, palette='coolwarm')    
f.tight_layout()



#%% #PairGrid example
diagnosis = df['diagnosis']
mean_cols = [col for col in df.columns if 'mean' in col]
meandf = pd.concat([diagnosis,df[mean_cols]], axis=1)

meandf.plot(logy=True)

g = s.PairGrid(meandf, hue="diagnosis")
g.map_diag(plt.hist)
g.map_offdiag(plt.scatter)
g.add_legend();



#%% ********** NEW RANDOM FOREST METHOD **********************
#--------------------------------------------------------------------------------------#
#--------------------------------------------------------------------------------------#
# Train Random Forest
np.random.seed(10)

traindf, testdf = train_test_split(df, test_size = 0.3)

x_train = traindf[x]
y_train = traindf[y]

x_test = testdf[x]
y_test = testdf[y]

forest = RandomForestClassifier(n_estimators=1000)
fit = forest.fit(x_train, y_train)
accuracy = fit.score(x_test, y_test)
predict = fit.predict(x_test)
cmatrix = confusion_matrix(y_test, predict)

#--------------------------------------------------------------------------------------#
# Perform k fold cross-validation


print ('Accuracy of Random Forest: %s' % "{0:.2%}".format(accuracy))

# Cross_Validation
v = cross_val_score(fit, x_train, y_train, cv=10)
for i in range(10):
    print('Cross Validation Score: %s'%'{0:2%}'.format(v[i,]))
    
#%%
#  Visualize Random Forest Confusion Matrix
plt.rcParams['figure.figsize']=(12,8)
ax = plt.axes()
s.heatmap(cmatrix, annot=True, fmt='d', ax=ax, cmap='coolwarm', annot_kws={"size": 30})
ax.set_title('Random Forest Confusion Matrix')


#%% **-## Not Working ##-**

# Random Forest ROC Curve
# calculate the fpr and tpr for all thresholds of the classification
probs = forest.predict_proba(x_test)
preds = probs[:,1]
fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
roc_auc = metrics.auc(fpr, tpr)

# method I: plt
import matplotlib.pyplot as plt
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

#%%Feature importances

importances = forest.feature_importances_
indices = np.argsort(importances)[::-1]

print("Feature ranking:")
for f in range(traindf[x].shape[1]):
    print("feature %s (%f)" % (list(traindf[x])[f], importances[indices[f]]))

feat_imp = pd.DataFrame({'Feature':list(traindf[x]),
                        'Gini importance':importances[indices]})
s.set_style('whitegrid')
ax = s.barplot(x='Gini importance', y='Feature', data=feat_imp)
ax.set(xlabel='Gini Importance')
plt.show()
#%% <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>|

#---------------------------------------------------------------------------------------#
# Train Support Vector Machine ---------------------------------------------------------#
#---------------------------------------------------------------------------------------#

np.random.seed(10)

traindf, testdf = train_test_split(df_norm, test_size = 0.3)

x_train = traindf[x]
y_train = traindf[y]

x_test = testdf[x]
y_test = testdf[y]

svmf = svm.SVC()
svm_fit = svmf.fit(x_train, y_train)
accuracy = svm_fit.score(x_test, y_test)
predict = svm_fit.predict(x_test)
svm_cm = confusion_matrix(y_test, predict)

#--------------------------------------------------------------------------------------#
# Perform k fold cross-validation
print ('Accuracy of Support Vector Machine: %s' % "{0:.2%}".format(accuracy))

# Cross_Validation
v = cross_val_score(svm_fit, x_train, y_train, cv=10)
for i in range(10):
    print('Cross Validation Score: %s'%'{0:2%}'.format(v[i,]))


#%% 
#   Visualize SVM Confusion Matrix
plt.rcParams['figure.figsize']=(12,8)
ax = plt.axes()
s.heatmap(svm_cm, annot=True, fmt='d', ax=ax, cmap="YlGnBu", annot_kws={"size": 30})
ax.set_title('Support Vector Machine Confusion Matrix')


#%% <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>|

#---------------------------------------------------------------------------------------#
# Train MLPClassifier ------------------------------------------------------------------#
#---------------------------------------------------------------------------------------#
np.random.seed(10)

traindf, testdf = train_test_split(df_norm, test_size = 0.3)

x_train = traindf[x]
y_train = traindf[y]

x_test = testdf[x]
y_test = testdf[y]

clf = MLPClassifier(solver='lbfgs', alpha=5, hidden_layer_sizes=(500,), random_state=10)
mlp_fit = clf.fit(x_train, y_train)
accuracy = mlp_fit.score(x_test, y_test)
predict = mlp_fit.predict(x_test)
mlp_cm = confusion_matrix(y_test, predict)

#--------------------------------------------------------------------------------------#
# Perform k fold cross-validation
print ('Accuracy of Multilayer Perceptron: %s' % "{0:.2%}".format(accuracy))

# Cross_Validation
v = cross_val_score(mlp_fit, x_train, y_train, cv=10)
for i in range(10):
    print('Cross Validation Score: %s'%'{0:2%}'.format(v[i,]))

#%%
#   Visualize MLP Confusion Matrix
plt.rcParams['figure.figsize']=(12,8)
ax = plt.axes()
s.heatmap(mlp_cm, annot=True, fmt='d', ax=ax, annot_kws={"size": 30})
ax.set_title('Multilayer Perceptron Confusion Matrix')

