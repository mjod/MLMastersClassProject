import pandas as pd
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
import numpy as np
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
from collections import Counter
from imblearn.over_sampling import SMOTE

def main():
    #Read in training set
    dataset = pd.read_csv("train.csv", nrows=100)
    dataset.hist()
    plt.show()
    # One hot encoding
    categorical_data = ['RESOURCE', 'MGR_ID', 'ROLE_ROLLUP_1', 'ROLE_ROLLUP_2', 'ROLE_DEPTNAME','ROLE_TITLE', 'ROLE_FAMILY_DESC', 'ROLE_FAMILY', 'ROLE_CODE']
    dataset = pd.get_dummies( dataset, columns = categorical_data )
    
    #Cross Validation
    array = dataset.values
    X = array[:,1:]
    Y = array[:,0]
    validation_size = 0.20
    seed = 7
    X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)


    # Use Synthetic Minority Over-sampling Technique to even imbalance of classes
    print('Original training class count {}'.format(Counter(Y_train)))
    sm = SMOTE(random_state=seed)
    X_train_resampled, Y_train_resampled = sm.fit_sample(X_train, Y_train)
    print('Resampled training class count {}'.format(Counter(Y_train_resampled)))
    print('Total # of samples after using SMOTE: {}'.format(X_train_resampled.shape[0]))
          
    # Test options and evaluation metric
    seed = 7
    scoring = 'accuracy'

    # Spot Check Algorithms
    models = []
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('DT', DecisionTreeClassifier()))
    models.append(('NB', GaussianNB()))
    models.append(('RFC', RandomForestClassifier()))
    models.append(('GBC', GradientBoostingClassifier()))
    
    # evaluate each model in turn
    results = []
    names = []
    for name, model in models:
        kfold = model_selection.KFold(n_splits=10, random_state=seed)
        cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
        results.append(cv_results)
        names.append(name)
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)
        model.fit(X_train, Y_train)
        predictions = model.predict(X_validation)
        print(accuracy_score(Y_validation, predictions))
        print(confusion_matrix(Y_validation, predictions))
        print(classification_report(Y_validation, predictions))
        print(roc_auc_score(Y_validation, predictions))
        print(matthews_corrcoef(Y_validation, predictions))
    
if __name__ == '__main__':
    main()