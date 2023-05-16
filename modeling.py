# import data manipulation tools
import numpy as np
import pandas as pd
# import data wrangling functions
import wrangle as w
# import nlp word counting functions
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
# import classification models
from sklearn.tree import DecisionTreeClassifier
# import classification metrics
from sklearn.metrics import accuracy_score
# ignore warning mesages
import warnings
warnings.filterwarnings("ignore")

# ====================================================================

def get_model_baseline(train, validate):
    '''
    This will display the metrics for the baseline model using the mode of the dataset
    '''
    # get the mode of the training dataset
    accuracy = train.language.value_counts(normalize=True)[0]
    # get the mode of the validation dataset
    acc_validate = validate.language.value_counts(normalize=True)[0]
    # create a dataframe to store the metrics
    metric_df = pd.DataFrame(data=[
        {
            # add the model metrics to the metrics_df
            'model' : 'baseline',
            'accuracy' : accuracy,
            'acc-validate' : acc_validate,
            'difference' : acc_validate - accuracy,
        }
    ])
    # return the baseline metrics
    return metric_df

# ====================================================================

def get_model_tree_tfidf_1gram_4depth(X_train, X_validate, y_train, y_validate):
    '''
    This will display metrics for a Descistion Tree classifier model using TfidfVectorizer
    with unigrams, and a max depth of 4.
    '''
    # create the TfidfVectorizer object with 1grams
    tfidf = TfidfVectorizer(ngram_range=(1, 1))
    # fit the tfid object, and transform the x_train dataset
    X_bow = tfidf.fit_transform(X_train)
    # transfrom the x_validate dataset
    X_val_bow = tfidf.transform(X_validate)
    # create a decision tree model with max depth 4
    tree = DecisionTreeClassifier(max_depth=4)
    # fit the decision tree on the training data
    tree.fit(X_bow, y_train)
    # get the accuracy score for the training data
    accuracy = tree.score(X_bow, y_train)
    # get the accuracy score for the validation data
    acc_validate = tree.score(X_val_bow, y_validate)
    # display the model metrics
    # display the model name and perameters
    print('Model : Decistion_Tree : TfidfVectorizer : 1gram : 4_max_depth')
    # display the accuracy on the training data
    print(f'Train accuracy : {accuracy}'),
    # display the accuracy on the validation data
    print(f'Validation accuracy : {acc_validate}'),
    # display the accuracy difference
    print(f'Difference : {acc_validate - accuracy}')
    # return the model for potential use on test data
    return tree, tfidf

# ====================================================================

def get_model_tree_tfidf_1gram_6depth(X_train, X_validate, y_train, y_validate):
    '''
    This will display metrics for a Descistion Tree classifier model using TfidfVectorizer
    with unigrams, and a max depth of 6.
    '''
    # create the TfidfVectorizer object with 1grams
    tfidf = TfidfVectorizer(ngram_range=(1, 1))
    # fit the tfid object, and transform the x_train dataset
    X_bow = tfidf.fit_transform(X_train)
    # transfrom the x_validate dataset
    X_val_bow = tfidf.transform(X_validate)
    # create a decision tree model with max depth 6
    tree = DecisionTreeClassifier(max_depth=6)
    # fit the decision tree on the training data
    tree.fit(X_bow, y_train)
    # get the accuracy score for the training data
    accuracy = tree.score(X_bow, y_train)
    # get the accuracy score for the validation data
    acc_validate = tree.score(X_val_bow, y_validate)
    # display the model metrics
    # display the model name and perameters
    print('Model : Decistion_Tree : TfidfVectorizer : 1gram : 6_max_depth')
    # display the accuracy on the training data
    print(f'Train accuracy : {accuracy}'),
    # display the accuracy on the validation data
    print(f'Validation accuracy : {acc_validate}'),
    # display the accuracy difference
    print(f'Difference : {acc_validate - accuracy}')
    # return the model for potential use on test data
    return tree, tfidf

# ====================================================================

def get_model_tree_cv_1gram_4depth(X_train, X_validate, y_train, y_validate):
    '''
    This will display metrics for a Descistion Tree classifier model using CountVectorizer
    with unigrams, and a max depth of 4.
    '''
    # create the CountVectorizer object with 1grams
    cv = CountVectorizer(ngram_range=(1, 1))
    # fit the tfid object, and transform the x_train dataset
    X_bow = cv.fit_transform(X_train)
    # transfrom the x_validate dataset
    X_val_bow = cv.transform(X_validate)
    # create a decision tree model with max depth 4
    tree = DecisionTreeClassifier(max_depth=4)
    # fit the decision tree on the training data
    tree.fit(X_bow, y_train)
    # get the accuracy score for the training data
    accuracy = tree.score(X_bow, y_train)
    # get the accuracy score for the validation data
    acc_validate = tree.score(X_val_bow, y_validate)
    # display the model metrics
    # display the model name and perameters
    print('Model : Decistion_Tree : CountVectorizer : 1gram : 4_max_depth')
    # display the accuracy on the training data
    print(f'train accuracy : {accuracy}'),
    # display the accuracy on the validation data
    print(f'validation accuracy : {acc_validate}'),
    # display the accuracy difference
    print(f'difference : {acc_validate - accuracy}')
    # return the model for potential use on test data
    return tree, cv

# ====================================================================

def get_model_test_tfifd_1(X_train, y_train, X_test, y_test, model, vectorizer):
    '''
    This will use the passed classification model and vectorizer on the test dataset
    '''
    # use the vectorizer to transfor the test data
    X_test_bow = vectorizer.transform(X_test)
    # use the model to get an accuracy score on the test data
    acc_test = model.score(X_test_bow, y_test)
    # display the model metrics
    # display the model info
    print('Test Dataset')
    print('Model : Decistion_Tree : TfidfVectorizer : 1gram : 4_max_depth')
    # display the model accuracy score
    print(f'Test accuracy : {acc_test}')