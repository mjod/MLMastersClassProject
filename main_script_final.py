# -*- coding: utf-8 -*-

import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from imblearn.over_sampling import SMOTE
from sklearn import model_selection
from sklearn.utils import shuffle
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import matthews_corrcoef
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier

"""
Evaluation of Performance Measures on Classifiers for Amazon Employee Access

We examine different ways to evaluate and manipulate the dataset for classifying 
users access needs to a system. Detecting when a user should be given or denied 
access to a database or system within their company would remove the need for an 
administrator to allocate the permission each time. We will build upon the previous 
study and this time evaluate on different measures to decide the best outcome. 
We will be using Naive​ ​Bayes,​ ​k-Nearest​ ​Neighbor,​ Decision​ ​Tree, Random​ ​Forest​
​and​ ​Gradient​ ​Boosting as our classifiers to help produce our measures. 
The performance measures we will be evaluating include accuracy, precision, 
recall, F-score, Area Under the Curve (AUC), False Positive and False Negative Rate,
and Matthews Correlation Coefficient (MCC). We will also be comparing how an 
oversampled dataset, SMOTE performs against an undersampled set. 
@author	Ian Ferringer
@author  Matthew O’Donnell 
"""
    
#Set constants
test_size = 0.20
seed = 7
bar_width = 0.35
opacity = 0.8
offset = 0.5

#Globals
categorical_data = ['ROLE_ROLLUP_1','ROLE_FAMILY']
accuracies = []
precisions = []
recalls = []
f1s = []
rocs = []
mccs = []
fprs = []
fnrs = []
scores = []

def one_hot_encode(dataset, cols):
    print "Current # of features:", len(dataset.columns.values)
    dataset = pd.get_dummies(dataset, columns = cols)
    print "Current # of features:", len(dataset.columns.values)
    return dataset

def k_nearest_neighbors_classifier():
    return KNeighborsClassifier()

def decision_tree_classifier():
    return DecisionTreeClassifier()

def naive_bayes_classifier():
    return GaussianNB()

def random_forest_classifier():
    return RandomForestClassifier()

def gradient_boosting_classifier():
    return GradientBoostingClassifier()

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion Matrix',
                          cmap=plt.cm.Blues):

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def raw_dataset(dataset):
    # We want to use the 1st attribute to the last attribute
    X = dataset.values[:,1:]
    # The 0th attribute is the target attribute
    Y = dataset.values[:,0]
    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=test_size, random_state=seed, stratify=Y)
    
    print('Original training class count {}'.format(Counter(Y_train)))
    print('Original test class count {}'.format(Counter(Y_test)))
    
    return X_train, X_test, Y_train, Y_test

def smote_dataset(dataset):
    # We want to use the 1st attribute to the last attribute
    X = dataset.values[:,1:]
    # The 0th attribute is the target attribute
    Y = dataset.values[:,0]
    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=test_size, random_state=seed, stratify=Y)
    
    print('Original training class count {}'.format(Counter(Y_train)))
    
    # Use Synthetic Minority Over-sampling Technique to even imbalance of classes
    sm = SMOTE(random_state=seed)
    X_train_resampled, Y_train_resampled = sm.fit_sample(X_train, Y_train)
    
    print('Resampled training class count {}'.format(Counter(Y_train_resampled)))
    print('Resampled test class count {}'.format(Counter(Y_test)))
    
    return X_train_resampled, X_test, Y_train_resampled, Y_test

def smote_balanced_dataset(dataset):
    print "Current Split by Classification:"
    print(dataset.groupby('ACTION').size())
    #Seperate out by classification types
    denied_set =  shuffle(dataset[dataset['ACTION'] == 0])
    approved_set =  shuffle(dataset[dataset['ACTION'] == 1])
    #get minimum number of columns between the two sets
    val=np.minimum(denied_set.shape, approved_set.shape)[0]
    test_set = denied_set[:val/2].append(approved_set[:val/2], ignore_index=True)
    X_test = test_set.values[:,1:]
    Y_test = test_set.values[:,0]
    train_set = denied_set[val/2:].append(approved_set[val/2:], ignore_index=True)
    print "Test Set Split by ACTION:"
    print(test_set.groupby('ACTION').size())
    print "Train Set Split by ACTION:"
    print(train_set.groupby('ACTION').size())
    
    # Use Synthetic Minority Over-sampling Technique to even imbalance of classes
    sm = SMOTE(random_state=seed)
    X_train_resampled, Y_train_resampled = sm.fit_sample(train_set.values[:,1:], train_set.values[:,0])
    print('Resampled training class count {}'.format(Counter(Y_train_resampled)))
    print('Resampled test class count {}'.format(Counter(Y_test)))

    return X_train_resampled, X_test, Y_train_resampled, Y_test

def undersampled_dataset(dataset):
    #Seperate out by classification types
    denied_set =  dataset[dataset['ACTION'] == 0]
    approved_set =  dataset[dataset['ACTION'] == 1]
    #get minimum number of columns between the two sets
    val=np.minimum(denied_set.shape, approved_set.shape)[0]
    dataset = denied_set.sample(n=val).append(approved_set.sample(n=val), ignore_index=True)
    # We want to use the 1st attribute to the last attribute
    X = dataset.values[:,1:]
    # The 0th attribute is the target attribute
    Y = dataset.values[:,0]
    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=test_size, random_state=seed, stratify=Y)

    print('unsampled training class count {}'.format(Counter(Y_train)))
    print('unsampled test class count {}'.format(Counter(Y_test)))
    return X_train, X_test, Y_train, Y_test

def train_test_evaluate(X_train, X_test, Y_train, Y_test, models):
    #Arrays to hold evaluation results
    accuracy = []
    precision = []
    recall = []
    f1 = []
    roc = []
    mcc = []
    fpr = []
    fnr = []
    score = []
    weight = float(1)/11
    idx = 0
    for name, model in models:
        print "#######################", name,"#######################"
        
        #Fit model and predict
        print("training model")
        model.fit(X_train, Y_train)
        
        print("testing model")
        predictions = model.predict(X_test)
        
        print("evaluating model")
        #Compute values for confusion matrix
        tn, fp, fn, tp = confusion_matrix(Y_test, predictions).ravel()
        
        #Add each model's results to the corresponding evaluation metric array
        accuracy.append(accuracy_score(Y_test, predictions))
        precision.append(precision_score(Y_test, predictions, average=None))
        print(classification_report(Y_test, predictions))
        recall.append(recall_score(Y_test, predictions, average=None))
        f1.append(f1_score(Y_test, predictions, average=None))
        roc.append(roc_auc_score(Y_test, predictions))
        mcc.append(matthews_corrcoef(Y_test, predictions))
        fpr.append(float(fp)/(fp+tn))
        fnr.append(float(fn)/(fn+tp))        
        
        score.append(float(weight)*((accuracy[idx])+(precision[idx][0])+(precision[idx][1])+
                           (recall[idx])[0]+(recall[idx][1])+
                           (f1[idx])[0]+(f1[idx][1])+
                           (roc[idx])+(mcc[idx])+(1-fpr[idx])+(1-fnr[idx])))
        print score[idx]
        #class names
        class_names = ['denied', 'allowed']
        np.set_printoptions(precision=2)
        cm = confusion_matrix(Y_test, predictions)
        print cm
        # Plot non-normalized confusion matrix
        plt.figure()
        plot_confusion_matrix(cm, classes=class_names,
                              title='Confusion matrix, without normalization')
        plt.show()
        idx = idx + 1
    
    accuracies.append(accuracy)
    precisions.append(precision)
    recalls.append(recall)
    f1s.append(f1)
    rocs.append(roc)
    mccs.append(mcc)
    fprs.append(fpr)
    fnrs.append(fnr)
    scores.append(score)
    
def plot_accuracies(dataset_names, model_names):
    num_datasets = np.arange(len(dataset_names))
    index = np.arange(len(model_names))
    
    plt.subplots()
    for dataset_index in num_datasets:
        plt.bar(index + bar_width*dataset_index, accuracies[dataset_index], bar_width, alpha=opacity, label=dataset_names[dataset_index])
    plt.xlabel('Models')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Comparison')
    plt.xticks(index + bar_width*offset, model_names)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.show()
    
def plot_precisions(dataset_names, model_names):
    num_datasets = np.arange(len(dataset_names))
    index = np.arange(len(model_names))
    
    #Plot for Denied Precision
    plt.subplots()
    for dataset_index in num_datasets:
        plt.bar(index + bar_width*dataset_index, np.array(precisions[dataset_index])[:,0], bar_width, alpha=opacity, label=dataset_names[dataset_index])
    plt.xlabel('Models')
    plt.ylabel('Precision')
    plt.title('Denied Precision Comparison')
    plt.xticks(index + bar_width*offset, model_names)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.show()
    
    #Plot for Allowed Precision
    plt.subplots()
    for dataset_index in num_datasets:
        plt.bar(index + bar_width*dataset_index, np.array(precisions[dataset_index])[:,1], bar_width, alpha=opacity, label=dataset_names[dataset_index])
    plt.xlabel('Models')
    plt.ylabel('Precision')
    plt.title('Allowed Precision Comparison')
    plt.xticks(index + bar_width*offset, model_names)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.show()
    
def plot_recall(dataset_names, model_names):
    num_datasets = np.arange(len(dataset_names))
    index = np.arange(len(model_names))
    
    #Plot for Denied Recall
    plt.subplots()
    for dataset_index in num_datasets:
        plt.bar(index + bar_width*dataset_index, np.array(recalls[dataset_index])[:,0], bar_width, alpha=opacity, label=dataset_names[dataset_index])
    plt.xlabel('Models')
    plt.ylabel('Recall')
    plt.title('Denied Recall Comparison')
    plt.xticks(index + bar_width*offset, model_names)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.show()
    
    #Plot for Allowed Recall
    plt.subplots()
    for dataset_index in num_datasets:
        plt.bar(index + bar_width*dataset_index, np.array(recalls[dataset_index])[:,1], bar_width, alpha=opacity, label=dataset_names[dataset_index])
    plt.xlabel('Models')
    plt.ylabel('Recall')
    plt.title('Allowed Recall Comparison')
    plt.xticks(index + bar_width*offset, model_names)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.show()
    
def plot_f1(dataset_names, model_names):
    num_datasets = np.arange(len(dataset_names))
    index = np.arange(len(model_names))
    
    #Plot for Denied F1-Score
    plt.subplots()
    for dataset_index in num_datasets:
        plt.bar(index + bar_width*dataset_index, np.array(f1s[dataset_index])[:,0], bar_width, alpha=opacity, label=dataset_names[dataset_index])
    plt.xlabel('Models')
    plt.ylabel('F1-Score')
    plt.title('Denied F1-Score Comparison')
    plt.xticks(index + bar_width*offset, model_names)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.show()
    
    #Plot for Allowed F1-Score
    plt.subplots()
    for dataset_index in num_datasets:
        plt.bar(index + bar_width*dataset_index, np.array(f1s[dataset_index])[:,1], bar_width, alpha=opacity, label=dataset_names[dataset_index])
    plt.xlabel('Models')
    plt.ylabel('F1-Score')
    plt.title('Allowed F1-Score Comparison')
    plt.xticks(index + bar_width*offset, model_names)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.show()
    
def plot_roc(dataset_names, model_names):
    num_datasets = np.arange(len(dataset_names))
    index = np.arange(len(model_names))
    
    plt.subplots()
    for dataset_index in num_datasets:
        plt.bar(index + bar_width*dataset_index, rocs[dataset_index], bar_width, alpha=opacity, label=dataset_names[dataset_index])
    plt.xlabel('Models')
    plt.ylabel('ROC_AUC Score')
    plt.title('Area Under ROC Curve Comparison')
    plt.xticks(index + bar_width*offset, model_names)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.show()
    
def plot_mcc(dataset_names, model_names):
    num_datasets = np.arange(len(dataset_names))
    index = np.arange(len(model_names))
    
    plt.subplots()
    for dataset_index in num_datasets:
        plt.bar(index + bar_width*dataset_index, mccs[dataset_index], bar_width, alpha=opacity, label=dataset_names[dataset_index])
    plt.xlabel('Models')
    plt.ylabel('MCC Score')
    plt.title('Matthews Correlation Coefficient Comparison')
    plt.xticks(index + bar_width*offset, model_names)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.show()
    
def plot_fpr(dataset_names, model_names):
    num_datasets = np.arange(len(dataset_names))
    index = np.arange(len(model_names))
    
    plt.subplots()
    for dataset_index in num_datasets:
        plt.bar(index + bar_width*dataset_index, fprs[dataset_index], bar_width, alpha=opacity, label=dataset_names[dataset_index])
    plt.xlabel('Models')
    plt.ylabel('FPR')
    plt.title('False Positive (Allow) Rate Comparison')
    plt.xticks(index + bar_width*offset, model_names)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.show()
    
def plot_fnr(dataset_names, model_names):
    num_datasets = np.arange(len(dataset_names))
    index = np.arange(len(model_names))
    
    plt.subplots()
    for dataset_index in num_datasets:
        plt.bar(index + bar_width*dataset_index, fnrs[dataset_index], bar_width, alpha=opacity, label=dataset_names[dataset_index])
    plt.xlabel('Models')
    plt.ylabel('FNR')
    plt.title('False Negative (Deny) Rate Comparison')
    plt.xticks(index + bar_width*offset, model_names)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.show()

def plot_score(dataset_names, model_names):
    num_datasets = np.arange(len(dataset_names))
    index = np.arange(len(model_names))
    
    plt.subplots()
    for dataset_index in num_datasets:
        plt.bar(index + bar_width*dataset_index, scores[dataset_index], bar_width, alpha=opacity, label=dataset_names[dataset_index])
    plt.xlabel('Models')
    plt.ylabel('Score')
    plt.title('Overall Score Evaluation')
    plt.xticks(index + bar_width*offset, model_names)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.show()

def main():
    #Read in training set
    dataset = pd.read_csv("train.csv")
    dataset = one_hot_encode(dataset, categorical_data)

    #Classification Algorithms
    models = []
    models.append(('KNN', k_nearest_neighbors_classifier()))
    models.append(('DT', decision_tree_classifier()))
    models.append(('NB', naive_bayes_classifier()))
    models.append(('RFC', random_forest_classifier()))
    models.append(('GBC', gradient_boosting_classifier()))
    model_names = np.array(models)[:,0]
    
    #Datasets to test
    dataset_names = []
    dataset_names.append('Raw')
    dataset_names.append('SMOTE')
    dataset_names.append('SMOTE')
    dataset_names.append('UnderSampled')
    
    #Perform training, testing, and evaluation of each dataset over all the models  
    print "####################### Using Raw Dataset #######################"
    X_train, X_test, Y_train, Y_test = raw_dataset(dataset)
    train_test_evaluate(X_train, X_test, Y_train, Y_test, models)
    
    print "####################### Using SMOTE Dataset #######################"
    X_train, X_test, Y_train, Y_test = smote_dataset(dataset)
    train_test_evaluate(X_train, X_test, Y_train, Y_test, models)
    
    print "####################### Using SMOTE Balanced Test Dataset #######################"
    X_train, X_test, Y_train, Y_test = smote_balanced_dataset(dataset)
    train_test_evaluate(X_train, X_test, Y_train, Y_test, models)
    
    print "####################### Using Undersampled Dataset #######################"
    X_train, X_test, Y_train, Y_test = undersampled_dataset(dataset)
    train_test_evaluate(X_train, X_test, Y_train, Y_test, models)
    
    #Plot evaluation metrics
    plot_accuracies(dataset_names, model_names)
    plot_precisions(dataset_names, model_names)
    plot_recall(dataset_names, model_names)
    plot_f1(dataset_names, model_names)
    plot_roc(dataset_names, model_names)
    plot_mcc(dataset_names, model_names)
    plot_fpr(dataset_names, model_names)
    plot_fnr(dataset_names, model_names)
    plot_score(dataset_names, model_names)
    

if __name__ == '__main__':
    main()