# Breast-Cancer-Prediction
<center><h4> Predicting Diagnosis of Breast Cancer Mass with Random Forest, Multilayer Perceptron, and Support Vector Machine </h4></center>
<center><h6> author: Neil Kutty </h6></center>
View ipython notebook: 
(https://github.com/sampsonsimpson/Breast-Cancer-Prediction/blob/master/Breast%20Cancer%20Data%20Prediction.ipynb)
<br><h6>Synopsis</h6><br>
This project attempts to predict if a breast cancer mass is Malignant or Benign based on 30 features of the cell nuclei. 

Including all available predictor versions in our model <b>(<i>Mean, Standard Error, and Worst</i>)</b> for the 10 core predictors (<b>Radius, Texture, Perimeter, Area, Smoothness, Compactness, Concavity, Concave Points, Symmetry, and Fractal Dimension</b>) we achieve a ~99% accuracy with Multilayer Perceptron Classification.

<h6> Dataset Description </h6>
<p> source: https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29<br>
source: https://www.kaggle.com/uciml/breast-cancer-wisconsin-data<br>
1st place submission: https://www.kaggle.com/buddhiniw/d/uciml/breast-cancer-wisconsin-data/breast-cancer-prediction<br>

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

a) radius (mean of distances from center to points on the perimeter)<br> 
b) texture (standard deviation of gray-scale values)<br> 
c) perimeter<br> 
d) area<br> 
e) smoothness (local variation in radius lengths)<br> 
f) compactness (perimeter^2 / area - 1.0)<br> 
g) concavity (severity of concave portions of the contour)<br> 
h) concave points (number of concave portions of the contour)<br> 
i) symmetry<br> 
j) fractal dimension ("coastline approximation" - 1)<br>

The mean, standard error and "worst" or largest (mean of the three largest values)
of these features were computed for each image, resulting in 30 features.
For instance, field 3 is Mean Radius, field 13 is Radius SE, field 23 is Worst Radius.

All feature values are recoded with four significant digits.

---


