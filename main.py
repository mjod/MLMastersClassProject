import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from imblearn.over_sampling import SMOTE
from pandas.plotting import scatter_matrix
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import matthews_corrcoef
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier

def generate_balanced_dataset(dataset):
    print "Current Split of Classification:"
    print(dataset.groupby('ACTION').size())

    #Seperate out by classification types
    denied_set =  dataset[dataset['ACTION'] == 0]
    approved_set =  dataset[dataset['ACTION'] == 1]
    #get minimum number of columns between the two sets
    val=np.minimum(denied_set.shape, approved_set.shape)[0]
    dataset = denied_set.sample(n=val).append(approved_set.sample(n=val), ignore_index=True)
    print "New Split of Classification:"
    print(dataset.groupby('ACTION').size())
    return dataset
    
def generate_smote_dataset(X_train, Y_train, seed):
    # Use Synthetic Minority Over-sampling Technique to even imbalance of classes
    print('Original training class count {}'.format(Counter(Y_train)))
    sm = SMOTE(random_state=seed)
    X_train_resampled, Y_train_resampled = sm.fit_sample(X_train, Y_train)
    print('Resampled training class count {}'.format(Counter(Y_train_resampled)))
    print('Total # of samples after using SMOTE: {}'.format(X_train_resampled.shape[0]))
    return X_train_resampled, Y_train_resampled

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

def plot_histogram(dataset):
    dataset.hist()
    plt.show()

def plot_scatter_matrix(dataset):
    scatter_matrix(dataset)
    plt.show()
    
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion Matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

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
    
def main():
    #Read in training set
    dataset = pd.read_csv("train.csv")#, nrows=100)

    #Generate a balanced dataset
    dataset = generate_balanced_dataset(dataset)
    
    #Plot histagram of each resource
    plot_histogram(dataset)
    
    # scatter plot matrix
    plot_scatter_matrix(dataset)
    
    #One hot encoding 'RESOURCE', 'MGR_ID', 
    categorical_data = ['ROLE_ROLLUP_1', 'ROLE_ROLLUP_2', 'ROLE_DEPTNAME',
                        'ROLE_TITLE', 'ROLE_FAMILY_DESC', 'ROLE_FAMILY', 'ROLE_CODE']
    dataset = one_hot_encode(dataset, categorical_data)
    
    #Cross Validation
    #Not using resource or mgr_id becuase previous researches proved it had low variance
    X = dataset.values[:,3:]
    Y = dataset.values[:,0]
    test_size = 0.20
    seed = 7
    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=test_size, 
                                                                        random_state=seed, stratify=Y)
    
    #Smote Balance Set
    #X_train, Y_train = generate_smote_dataset(X_train, Y_train, seed)

    #Classification Algorithms
    models = []
    models.append(('KNN', k_nearest_neighbors_classifier()))
    models.append(('DT', decision_tree_classifier()))
    models.append(('NB', naive_bayes_classifier()))
    models.append(('RFC', random_forest_classifier()))
    models.append(('GBC', gradient_boosting_classifier()))
    
    # Test options and evaluation metric
    scoring = 'accuracy'
    
    # evaluate each model in turn
    results = []
    names = []
    for name, model in models:
        print "#######################", name,"#######################"
        #Cross validation test
        kfold = model_selection.KFold(n_splits=3, random_state=seed)
        cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
        results.append(cv_results)
        names.append(name)
        msg = "%s cross validation results: mean = %f | std = %f" % (name, cv_results.mean(), cv_results.std())
        print(msg)
        
        #Fit model and predict
        model.fit(X_train, Y_train)
        predictions = model.predict(X_test)
        print("accuracy: %f" % (accuracy_score(Y_test, predictions)))

        print("classification_report:")
        print(classification_report(Y_test, predictions))
        print("roc_auc_score: %f" % (roc_auc_score(Y_test, predictions)))
        print("matthews_corrcoef: %f" % (matthews_corrcoef(Y_test, predictions)))
        
        #class names
        class_names = ['denied', 'allowed']
        np.set_printoptions(precision=2)
        cm = confusion_matrix(Y_test, predictions)


        # Plot non-normalized confusion matrix
        plt.figure()
        plot_confusion_matrix(cm, classes=class_names,
                              title='Confusion matrix, without normalization')
        
        # Plot normalized confusion matrix
        plt.figure()
        plot_confusion_matrix(cm, classes=class_names, normalize=True,
                              title='Normalized confusion matrix')
        
        plt.show()
    
if __name__ == '__main__':
    main()