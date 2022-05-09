#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 16:15:44 2022

@author: laurenwilkes
"""

import pandas as pd

df = pd.read_csv("~/Downloads/star_classification.csv")

import pandas as pd

import numpy as np 

import matplotlib.pyplot as plt

import sklearn

from sklearn.neural_network import MLPClassifier

from sklearn.neural_network import MLPRegressor


# Import necessary modules

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error

from math import sqrt

from sklearn.metrics import r2_score
from sklearn.metrics import classification_report,confusion_matrix

X = df.drop('class', axis = 1)
y = df['class']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=40)

from sklearn.neural_network import MLPClassifier

mlp_gs = MLPClassifier(max_iter=100)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

parameter_space = {
    'hidden_layer_sizes': [(10,),(100,),(50,)],
    'activation': ['tanh', 'relu', 'identity', 'logistic'],
}
from sklearn.model_selection import GridSearchCV
clf = GridSearchCV(mlp_gs, parameter_space, n_jobs=-1, cv=5)
clf.fit(X_train,y_train) # X is train samples and y is the corresponding labels

print('Best parameters found:\n', clf.best_params_)


mlp = MLPClassifier(hidden_layer_sizes=(50,), activation='tanh')

mlp.fit(X_train,y_train)


predict_train = mlp.predict(X_train)

predict_test = mlp.predict(X_test)



print(confusion_matrix(y_test,predict_test))
data = confusion_matrix(y_test,predict_test)
print(classification_report(y_test,predict_test))

import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

def plot_confusion_matrix(data, labels, output_filename):
    """Plot confusion matrix using heatmap.
 
    Args:
        data (list of list): List of lists with confusion matrix data.
        labels (list): Labels which will be plotted across x and y axis.
        output_filename (str): Path to output file.
 
    """
    sn.set(color_codes=True)
    plt.figure(1, figsize=(9, 6))
 
    plt.title("Confusion Matrix")
 
    sn.set(font_scale=1.4)
    ax = sn.heatmap(data, annot=True, cmap="YlGnBu", cbar_kws={'label': 'Scale'})
 
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
 
    ax.set(ylabel="True Label", xlabel="Predicted Label")
    plt.show()
 
    #plt.savefig(output_filename, bbox_inches='tight', dpi=300)
    plt.close()

# define labels
labels = ["Galaxy", "QSO", "STAR"]
 
# create confusion matrix
plot_confusion_matrix(data, labels, "confusion_matrix.png")

#Bayesian Neural Network
import numpy as np
from sklearn import datasets

import torch
import torch.nn as nn
import torch.optim as optim

import torchbnn as bnn

import matplotlib.pyplot as plt
%matplotlib inline

y_train[y_train== 'GALAXY'] = 0
y_train[y_train== 'QSO'] = 1
y_train[y_train== 'STAR'] = 2

y_test[y_test== 'GALAXY'] = 0
y_test[y_test== 'QSO'] = 1
y_test[y_test== 'STAR'] = 2

X = X_train
Y = y_train.to_numpy()
Y = Y.astype('float64')

x = torch.from_numpy(X).float()
y=Y
y = torch.from_numpy(Y).long()
x.shape, y.shape

model = nn.Sequential(
    bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=17, out_features=200),
    nn.ReLU(),
    bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=200, out_features=200),
    nn.ReLU(),
    bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=200, out_features=200),
    nn.ReLU(),
    bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=200, out_features=200),
    nn.ReLU(),
    bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=200, out_features=200),
    nn.ReLU(),
    bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=200, out_features=200),
    nn.ReLU(),
    bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=200, out_features=3),
)

ce_loss = nn.CrossEntropyLoss()
kl_loss = bnn.BKLLoss(reduction='mean', last_layer_only=False)
kl_weight = 0.01

optimizer = optim.Adam(model.parameters(), lr=0.001)

kl_weight = 0.1

for step in range(3000):
    pre = model(x)
    ce = ce_loss(pre, y)
    kl = kl_loss(model)
    cost = ce + kl_weight*kl
    
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()
    
_, predicted = torch.max(pre.data, 1)
total = y.size(0)
correct = (predicted == y).sum()
print('- Accuracy: %f %%' % (100 * float(correct) / total))
print('- CE : %2.2f, KL : %2.2f' % (ce.item(), kl.item()))

x_test= torch.from_numpy(X_test).float()

pre = model(x_test)
_, predicted = torch.max(pre.data, 1)
pred= pd.DataFrame(predicted)
pred = pred.astype(object).squeeze()

y_test = y_test.astype(int)
pred = pred.astype(int)
print(classification_report(y_test,pred))

#good results with 200 out features and 0.001 learning rate





















































###STATISTICS

from warnings import filterwarnings

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pymc3 as pm
import seaborn as sns
import sklearn
import theano
import theano.tensor as T

from sklearn import datasets
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale

floatX = theano.config.floatX

y_train.replace({'GALAXY':0, 'QSO':1, 'STAR':2}, inplace=True)

y_test.replace({'GALAXY':0, 'QSO':1, 'STAR':2}, inplace=True)

def construct_nn(ann_input, ann_output):
    n_hidden = 5
    
    ann_input = pm.Data("ann_input", X_train)
    ann_output = pm.Data("ann_output", y_train)

    # Initialize random weights between each layer
    init_1 = np.random.randn(X.shape[1], n_hidden).astype(floatX)
    init_2 = np.random.randn(n_hidden, n_hidden).astype(floatX)
    init_out = np.random.randn(n_hidden).astype(floatX)
     # Initialize random biases in each layer
    init_b_1 = np.random.randn(n_hidden).astype(floatX)
    init_b_2 = np.random.randn(n_hidden).astype(floatX)
    init_b_out = np.random.randn(1).astype(floatX)

    with pm.Model() as neural_network:
        # Weights from input to hidden layer
        weights_in_1 = pm.Normal('w_in_1', 0, sd=1,
                                 shape=(X.shape[1], n_hidden),
                                 testval=init_1)
        bias_1 = pm.Normal('b_1', mu=0, sd=1, shape=(n_hidden), testval=init_b_1)
        # Weights from 1st to 2nd layer
        weights_1_2 = pm.Normal('w_1_2', 0, sd=1,
                                shape=(n_hidden, n_hidden),
                                testval=init_2)
        bias_2 = pm.Normal('b_2', mu=0, sd=1, shape=(n_hidden), testval=init_b_2)
        # Weights from hidden layer to output
        weights_2_out = pm.Normal('w_2_out', 0, sd=1,
                                  shape=(n_hidden,),
                                  testval=init_out)
        bias_out = pm.Normal('b_out', mu=0, sd=1, shape=(1), testval=init_b_out)

        # Build neural-network using tanh activation function
        act_1 = pm.math.tanh(pm.math.dot(ann_input, weights_in_1) + bias_1)
        act_2 = pm.math.tanh(pm.math.dot(act_1, weights_1_2) + bias_2)
        act_out = pm.math.dot(act_2, weights_2_out) + bias_out
        sd = pm.HalfNormal('sd', sd=1)
        
        # Binary classification -> Bernoulli likelihood
        out = pm.Bernoulli(
            "out",
            act_out,
            observed=ann_output,
            total_size=y_train.shape[0],  # IMPORTANT for minibatches
        )                  
    return neural_network


neural_network = construct_nn(X_train, y_train)

from pymc3.theanof import MRG_RandomStreams, set_tt_rng

set_tt_rng(MRG_RandomStreams(42))

with neural_network:
    inference = pm.ADVI()
    approx = pm.fit(n=30000, method=inference)
    
plt.plot(-inference.hist, label="new ADVI", alpha=0.3)
plt.plot(approx.hist, label="old ADVI", alpha=0.3)
plt.legend()
plt.ylabel("ELBO")
plt.xlabel("iteration");

trace = approx.sample(draws=5000)

# We can get predicted probability from model
neural_network.out.distribution.p

# create symbolic input
x = T.matrix("X")
# symbolic number of samples is supported, we build vectorized posterior on the fly
n = T.iscalar("n")
# Do not forget test_values or set theano.config.compute_test_value = 'off'
x.tag.test_value = np.empty_like(X_train[:10])
n.tag.test_value = 100
_sample_proba = approx.sample_node(
    neural_network.out.distribution.p, size=n, more_replacements={neural_network["ann_input"]: x}
)
# It is time to compile the function
# No updates are needed for Approximation random generator
# Efficient vectorized form of sampling is used
sample_proba = theano.function([x, n], _sample_proba)

# Create bechmark functions
def production_step1():
    pm.set_data(new_data={"ann_input": X_test, "ann_output": Y_test}, model=neural_network)
    ppc = pm.sample_posterior_predictive(
        trace, samples=500, progressbar=False, model=neural_network
    )

    # Use probability of > 0.5 to assume prediction of class 1
    pred = ppc["out"].mean(axis=0) > 0.5


def production_step2():
    sample_proba(X_test, 500).mean(0) > 0.5
    
predict_test = sample_proba(X_test, 500).mean(0) > 0.5
    
print(classification_report(y_test,predict_test))



