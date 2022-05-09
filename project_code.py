import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import statistics
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn import tree
from sklearn.model_selection import GridSearchCV


data = pd.read_csv("star_classification.csv")
#obj_ID: drop ID: identifier
#rerun_ID: all observations have same value :: provides no information
data = data.drop(['obj_ID', 'rerun_ID'], axis=1)
variable_list = data.columns
#split into train and test
data_train, data_test = train_test_split(data, test_size=0.2, random_state=0)


#EXPLORATORY ANALYSIS

#classes:
len(data_train.loc[data_train["class"] == "GALAXY"])
len(data_train.loc[data_train["class"] == "STAR"])
len(data_train.loc[data_train["class"] == "QSO"])

#create histograms of observed values
def eda_hist(var):
    train_gal = data_train.loc[data_train["class"] == "GALAXY"][var]
    train_star = data_train.loc[data_train["class"] == "STAR"][var]
    train_qso = data_train.loc[data_train["class"] == "QSO"][var]
    
    plt.hist(train_gal, label = "Class = Galaxy", density = True, alpha=0.5, bins=30)
    plt.hist(train_star, label = "Class = Star", density = True, alpha=0.5, bins=30)
    plt.hist(train_qso, label = "Class = Qso", density = True, alpha=0.5, bins=30)
    
    plt.title("{} Observed Values".format(var))
    plt.xlabel("{}".format(var))
    plt.ylabel("Frequency")
    plt.legend(loc='upper right')
    plt.show()
    plt.gca().clear()

#print descriptive statistics
def eda_desc(var):
    train_gal = data_train.loc[data_train["class"] == "GALAXY"]
    train_star = data_train.loc[data_train["class"] == "STAR"]
    train_qso = data_train.loc[data_train["class"] == "QSO"]
    
    print("{} descriptive statistics:".format(var))
    print("   galaxy:", desc_metrics(train_gal, var))
    print("   star:  ", desc_metrics(train_star, var))
    print("   qso:   ", desc_metrics(train_qso, var))
    
#return min, max, mean, sd for a variable and dataset
def desc_metrics(data, var):
    var_max = max(data[var])
    var_min = min(data[var])
    var_mean = statistics.mean(data[var])
    var_sd = statistics.stdev(data[var])
    
    return [var_min, var_max, var_mean, var_sd]

#class: drop response for X
data_train_X = data_train.drop(['class'], axis=1)

#classification variables
features = data_train_X.columns

#output EDA
#for var in features:
    #eda_hist(var)
    #eda_desc(var)
    
#outlier detection
#3,4,7 producing -9999.0 minimum
data_train = data_train[data_train.u != -9999.0]
data_train = data_train[data_train.g != -9999.0]
data_train = data_train[data_train.z != -9999.0]

#correlation matrix
sn.heatmap(data_train.corr(), annot=True)
plt.show()

    

#MODELING

#NULL MODEL (baseline)

#classify majority class: GALAXY
    
#predict galaxy for all observations
test_y_null_pred = ["GALAXY"] * len(data_test)

#assess null performance: compare actual and predicted class
print("Confusion Matrix: Null (train)")
print(metrics.confusion_matrix(data_test["class"], test_y_null_pred))
print(metrics.classification_report(data_test["class"], test_y_null_pred))


#K-NEAREST NEIGHBORS

#knn model for provided k
def knn(k):
    knn_model = KNeighborsClassifier(n_neighbors=k)
    knn_model.fit(data_train[features], data_train["class"])

    #generate predictions
    test_y_knn_pred = knn_model.predict(data_test[features])
    return test_y_knn_pred

#selecting optimal k
F1_scores_knn = []
for k in range(1,100):
    test_y_knn_pred = knn(k)
    F1_scores_knn.append(metrics.f1_score(data_test["class"], test_y_knn_pred, average="weighted"))

#plot k
plt.plot(range(1,100,1), F1_scores_knn)
plt.title("KNN: k versus weighted F1")
plt.xlabel("k")
plt.ylabel("weighted F1")
k_opt = F1_scores_knn.index(max(F1_scores_knn)) + 1 #index starts a 0, k at 1
#optimal k = 10, based on max weighted F1 score

#assess optimal knn performance
test_y_knn_pred = knn(10)
print("Confusion Matrix: KNN")
print(metrics.confusion_matrix(data_test["class"], test_y_knn_pred))
print(metrics.classification_report(data_test["class"], test_y_knn_pred))


#SUPPORT VECTOR MACHINE (SVM)

clf = make_pipeline(StandardScaler(), GridSearchCV(estimator=SVC(), param_grid={'C': [1, 10], 'kernel': ('linear', 'rbf'), 'gamma': ('scale', 'auto')})) #gridsearch SVM
clf.fit(data_train[features], data_train["class"])

#generate predictions
test_y_svm_pred = clf.predict(data_test[features])

print(metrics.confusion_matrix(data_test["class"], test_y_svm_pred))
print(metrics.classification_report(data_test["class"], test_y_svm_pred)) 

# DECISION TREE

clf = tree.DecisionTreeClassifier(max_depth=4, max_leaf_nodes=16, min_samples_leaf=4, min_samples_split=4)
clf.fit(data_train[features], data_train["class"])

#generate predictions
test_y_tree_pred = clf.predict(data_test[features])

print("Confusion Matrix: Decision Tree")
print(metrics.confusion_matrix(data_test["class"], test_y_tree_pred))
print(metrics.classification_report(data_test["class"], test_y_tree_pred))

fig, axes = plt.subplots(figsize = (33, 33))
tree.plot_tree(clf, ax=axes, fontsize=10, filled=True)
fig.savefig('imagename.png')