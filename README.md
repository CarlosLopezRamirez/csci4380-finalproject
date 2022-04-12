# csci4380-finalproject

## Elise Karinshak, Carlos Lopez-Ramirez, Lauren Rose Wilkes

The stellar classification dataset contains 100,000 observations of space from the Sloan Digital Sky Survey. 
For each observation, features describing spectral characteristics (characteristics relating to wavelength, 
such as alpha, delta, redshift, etc; 16 in total) are recorded. With this data, we seek to classify stars, 
galaxies, and quasars. Such analysis is meaningful in applications of astrophysics and processing of astronomical data; 
successful classification can aid astronomers in understanding the makeup of the universe. As scientists create more 
powerful spectral detection tools, the analysis of their results becomes increasingly important. 

## Proposed Solution

First, we will process the dataset to prepare it for our analysis. We will split the dataset, designating ⅔ of the dataset 
for training and ⅓ for testing. This will allow us to build our models but also leave some data to evaluate on unseen data. 

As a first step in our analysis, we will conduct exploratory analysis of the dataset. This analysis will include handling 
outliers and missing data, observing levels of correlation between features, and feature selection techniques. We will also 
investigate any imbalance in the data and implement balancing techniques accordingly. We will also explore standardizing 
and centering the data. 

We will then apply modeling techniques to this classification question. Each team member will be responsible for a model 
we covered in class (as a baseline for comparison), as well as an additional model not covered for further exploration.

## Elise Karinshak

### kNN
A classic classification technique, utilizing the classes of nearby neighbors to predict class labels. 
However, a weakness of this technique is the computational costs, which will be significant with our 
large dataset.

### SVM
Determines an optimal hyperplane to separate the data in classification. This is an effective technique 
when the data has a clear margin of separation but may not work as well if the data has a lot of noise. 
We expect this model to perform better than kNN. 

## Lauren Wilkes

### Neural Network
A model discussed in class. Neural networks take input and transmit it through “neurons” in a series of 
layers to learn patterns in the data to ultimately produce an output. Neural networks have high flexibility, 
and we anticipate a good prediction which makes it an appealing choice of model. However, neural networks 
are harder to explain and can be computationally expensive to train. 

### Bayesian Neural Network
Bayesian neural networks offer an uncertainty quantification in addition to their prediction. This will 
be extremely useful as it allows researchers to evaluate how confident the model is in its prediction 
and accept or further investigate the prediction accordingly. This makes it a very useful model, 
although it is important to note that it still suffers from some of the downsides of a regular 
neural network including high computation costs and lower explainability. 

## Carlos Lopez Ramirez

### Decision Tree
A model we have discussed in class. This is a supervised model that uses cascading questions based on 
different attributes within the data to classify an observation. This has the advantage of being very 
easy to explain. However, it does not handle missing values well, and it can be computationally expensive 
to train. It also may not give as high prediction results as some more advanced models. 

### Random Forest
Creates different random decision trees and decorrelates the bagged trees, combining individual tree’s 
predictions to produce the final output. We expect this to perform better than standard decision trees. 
However, random forests are less easily interpretable and can be more computationally expensive. 

## Evaluation
To evaluate these models, we will compare metrics such as accuracy, precision, recall, and F-1 score to 
get a holistic understanding of performance. We will also measure the computing time of the models, 
as we are dealing with a large dataset. We will then assess the benefits and weaknesses of each model 
in the context of the problem, and ultimately provide recommendations.
