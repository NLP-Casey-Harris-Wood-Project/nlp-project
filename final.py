# =======================================================================================================
# Table of Contents START
# =======================================================================================================

'''
1. Orientation
2. Imports
3. vanilla
4. prepare
5. visual1
6. visual2
7. visual3
8. visual4
9. visual5
10. stat1
11. baseline
12. models
13. top3models
'''

# =======================================================================================================
# Table of Contents END
# Table of Contents TO Orientation
# Orientation START
# =======================================================================================================

'''
The purpose of this file is to create functions specifically for the 'final_report.ipynb'
file and anything else that may be of use.
'''

# =======================================================================================================
# Orientation END
# Orientation TO Imports
# Imports START
# =======================================================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from wordcloud import WordCloud
import nltk
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
import nltk.sentiment
import re
import unicodedata
import wrangle as w
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# =======================================================================================================
# Imports END
# Imports TO vanilla
# vanilla START
# =======================================================================================================

def vanilla():
    '''
    Returns the shape of the vanilla dataframe

    INPUT:
    NONE

    OUTPUT:
    vanilla.shape
    '''
    # read in the raw data from the local csv
    return pd.read_csv('repo.csv', index_col=0)

# =======================================================================================================
# vanilla END
# vanilla TO prepare
# prepare START
# =======================================================================================================

def prepare():
    '''
    Acquires the fully prepared version of the dataframe and returns it

    INPUT:
    NONE

    OUTPUT:
    repo_df = Fully prepared pandas dataframe
    '''
    # acquire and prep the dataset
    return w.post_explore_wrangle_github_repositories()

# =======================================================================================================
# prepare END
# prepare TO train_split
# train_split START
# =======================================================================================================

def train_split(df):
    '''
    Takes in a dataframe and splits the data into train, validate, 
    and test sets with 70%, 20%, 10% of data.

    INPUT:
    df = Pandas dataframe to be splitted

    OUTPUT:
    train = Pandas dataframe with 70% of original data
    validate = Pandas dataframe with 20% of original data
    test = Pandas dataframe with 10% of original data
    '''
    # get split off the test data
    train_validate, test = train_test_split(df,
                                            # set the random seed
                                            random_state=1349,
                                            # set the test dataset size
                                            train_size=0.9,
                                            # we want an even amount of python and html 
                                            # pages in each group
                                            stratify=df.language)
    # get the train and validate datasets
    train, validate = train_test_split(train_validate,
                                       # set the random seed
                                       random_state=1349,
                                       # set the training dataset size
                                       train_size=0.778,
                                       # we want an even amount of python and html 
                                       # pages in each group
                                       stratify=train_validate.language)
    # return the train, validate and test datasets
    return train, validate, test

# =======================================================================================================
# train_split END
# train_split TO get_words
# get_words START
# =======================================================================================================

def get_words(train):
    '''
    This will split the training dataset into groups containing text from the python
    and html language groups
    '''
    # get the words from the python language repos
    python_text = ' '.join(train[train.language == 'Python'
                                 ]['cleaned_readme_contents'].astype(str))
    # get the words from the html language repos
    html_text = ' '.join(train[train.language == 'HTML'
                               ]['cleaned_readme_contents'].astype(str))
    # return the two language groups
    return python_text, html_text

# =======================================================================================================
# get_words END
# get_words TO visual1
# visual1 START
# =======================================================================================================

def visual1(before_df, after_df):
    '''
    Shows the distribution of 'srchttps' per repository as a subplot 
    with 2 visuals that demonstrate the distribution before the outlier 
    removal and one after the outlier removal

    INPUT:
    NONE

    OUTPUT:
    visual = Subplot with 2 distribution plots, one before outlier removal, 
    one after outlier removal
    '''
    # Assuming you have a DataFrame named 'repos_df' that contains repository information
    python_df = before_df[before_df['language'] == 'Python']
    
    matched_repos = python_df[python_df['cleaned_readme_contents'].str.contains("srchttps")]
    matched_repos['frequency'] = matched_repos['cleaned_readme_contents'].str.count("srchttps")
    
    sorted_repos = matched_repos.sort_values('frequency', ascending=False)
    repo_freq_list = sorted_repos[['repo', 'frequency']].values.tolist()
    repos = [repo for repo, freq in repo_freq_list]
    frequencies = [freq for repo, freq in repo_freq_list]
    
    python_df_after = after_df[after_df['language'] == 'Python']
    matched_repos_after = python_df_after[python_df_after['cleaned_readme_contents'].\
                                          str.contains("srchttps")]
    matched_repos_after['frequency'] = matched_repos_after['cleaned_readme_contents'].\
                                        str.count("srchttps")
    
    sorted_repos_after = matched_repos_after.sort_values('frequency', ascending=False)
    repo_freq_list_after = sorted_repos_after[['repo', 'frequency']].values.tolist()
    repos_after = [repo for repo, freq in repo_freq_list_after]
    frequencies_after = [freq for repo, freq in repo_freq_list_after]
    
    # create visualization
    plt.style.use('ggplot')
    # create subplots
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].barh(repos, frequencies)
    axs[0].set_xlabel('Frequency')
    axs[0].set_ylabel('Repository')
    axs[0].set_title('Frequency of "srchttps" in Repositories (Before)')
    axs[0].set_yticks([])
    axs[1].barh(repos_after, frequencies_after)
    axs[1].set_xlabel('Frequency')
    axs[1].set_ylabel('Repository')
    axs[1].set_title('Frequency of "srchttps" in Repositories (After)')
    axs[1].set_yticks([])
    plt.tight_layout()
    plt.show()

# =======================================================================================================
# visual1 END
# visual1 TO visual2
# visual2 START
# =======================================================================================================

def visual2(train, python_text, html_text):
    '''
    Gets the distribution of unique words for both Python and HTML repositories

    INPUT:
    NONE

    OUTPUT:
    visual = Subplot with 2 distribution visuals of unique words, one for Python, one for HTML
    '''
#     python_df = train[train['language'] == 'Python']
#     html_df = train[train['language'] == 'HTML']
    
#     python_text = ' '.join(python_df['cleaned_readme_contents'])
#     python_text = python_text.lower()
    python_tokens = word_tokenize(python_text)
    
    python_freqdist = FreqDist(python_tokens)
    
#     html_text = ' '.join(html_df['cleaned_readme_contents'])
#     html_text = html_text.lower()
    html_tokens = word_tokenize(html_text)
    
    html_freqdist = FreqDist(html_tokens)
    
    python_word_freq = [(word, freq) for word, freq in python_freqdist.items() 
                        if word not in html_freqdist.keys()]
    sorted_python_word_freq = sorted(python_word_freq, key=lambda x: x[1], reverse=True)
    top_10_python_words = [pair[0] for pair in sorted_python_word_freq][:10]
    top_10_python_frequencies = [pair[1] for pair in sorted_python_word_freq][:10]
    html_word_freq = [(word, freq) for word, freq in 
                      html_freqdist.items() if word not in python_freqdist.keys()]
    
    sorted_html_word_freq = sorted(html_word_freq, key=lambda x: x[1], reverse=True)
    top_10_html_words = [pair[0] for pair in sorted_html_word_freq][:10]
    top_10_html_frequencies = [pair[1] for pair in sorted_html_word_freq][:10]
    
    data = {
        'Python Word': top_10_python_words,
        'Python Frequency': top_10_python_frequencies,
        'HTML Word': top_10_html_words,
        'HTML Frequency': top_10_html_frequencies
    }
    df = pd.DataFrame(data)
    
    df_python_sorted = df.sort_values(by='Python Frequency', ascending=False)
    df_html_sorted = df.sort_values(by='HTML Frequency', ascending=False)
    
    plt.style.use('ggplot')
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 8))
    axes[0].barh(df_python_sorted['Python Word'],\
                 df_python_sorted['Python Frequency'],\
                 color='tab:blue')
    axes[0].set_title('Top 10 Unique Words: Python')
    axes[0].set_xlabel('Frequency')
    axes[0].set_ylabel('Words')
    axes[1].barh(df_html_sorted['HTML Word'],\
                 df_html_sorted['HTML Frequency'],\
                 color='tab:orange')
    axes[1].set_title('Top 10 Unique Words: HTML')
    axes[1].set_xlabel('Frequency')
    axes[1].set_ylabel('Words')
    plt.tight_layout()
    plt.show()


    
# =======================================================================================================
# visual2 END
# visual2 TO visual3
# visual3 START
# =======================================================================================================

# def visual3(train, python_text, html_text):
#     '''
#     Creates wordclouds of most commonly used words across repositories for both Python and HTML

#     INPUT:
#     NONE

#     OUTPUT:
#     visual = Subplot with 2 wordclouds, one for Python, one for HTML
#     '''
    
#     unique_python_words = set(python_text.split())
#     unigram_python_img = WordCloud(background_color='white').\
#         generate(' '.join(unique_python_words))
    
#     unique_html_words = set(html_text.split())
#     unigram_html_img = WordCloud(background_color='white').\
#         generate(' '.join(unique_html_words))
    
#     fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#     axs[0].imshow(unigram_python_img)
#     axs[0].axis('off')
#     axs[0].set_title('Most Common Python Words')
#     axs[1].imshow(unigram_html_img)
#     axs[1].axis('off')
#     axs[1].set_title('Most Common HTML Words')
#     plt.tight_layout()
#     plt.show()

def visual3(train):
    '''
    This will display the mean average sentiment score for python and html repos
    '''
    # create sentiment analyzer object
    sia = nltk.sentiment.SentimentIntensityAnalyzer()
    # get sentiment scores for every repo in the training dataset
    train['compound_sentiment'] = train['cleaned_readme_contents'].apply(
        lambda x: sia.polarity_scores(x)['compound'])
    # get the average sentiment score for each language
    sentiments = train.groupby('language')['compound_sentiment'].mean()
    # display a horizontal bar plot of the avg sentiment scores
    sentiments.plot.barh()
    # add title
    plt.title('Python Has Slightly Higher Average Sentiment Score', size=16)
    # add axis labels
    plt.xlabel('Average Sentiment Score', size=16)
    plt.ylabel('Programming Language', size=16)
    # resize tick labels
    plt.xticks(size=14)
    plt.yticks(size=14)
    # show plot
    plt.show()

# =======================================================================================================
# visual3 END
# visual3 TO visual4
# visual4 START
# =======================================================================================================

def visual4(train, python_text, html_text):
    '''
    Returns a subplot of the distribution of bigrams for both Python and HTML repositories

    INPUT:
    NONE

    OUTPUT:
    visual = Subplot with 2 distribution visuals, one for Python, one for HTML
    '''
#     python_df = train[train['language'] == 'Python']
#     html_df = train[train['language'] == 'HTML']
#     python_text = ' '.join(python_df['cleaned_readme_contents'])
#     html_text = ' '.join(html_df['cleaned_readme_contents'])
    
    python_bigrams = list(nltk.bigrams(nltk.word_tokenize(python_text)))
    html_bigrams = list(nltk.bigrams(nltk.word_tokenize(html_text)))
    
    python_bigram_freqdist = FreqDist(python_bigrams)
    html_bigram_freqdist = FreqDist(html_bigrams)
    
    python_unique_bigrams = set(python_bigram_freqdist.keys()) - set(html_bigram_freqdist.keys())
    html_unique_bigrams = set(html_bigram_freqdist.keys()) - set(python_bigram_freqdist.keys())
    
    sorted_python_bigrams = sorted(python_unique_bigrams, key=lambda x: 
                                   python_bigram_freqdist[x], reverse=True)
    sorted_html_bigrams = sorted(html_unique_bigrams, key=lambda x: 
                                 html_bigram_freqdist[x], reverse=True)
    
    top_10_python_bigrams = sorted_python_bigrams[:10]
    python_bigram_frequencies = [python_bigram_freqdist[bigram] for 
                                 bigram in top_10_python_bigrams]
    top_10_html_bigrams = sorted_html_bigrams[:10]
    html_bigram_frequencies = [html_bigram_freqdist[bigram] for 
                               bigram in top_10_html_bigrams]
    
    plt.figure(figsize=(10, 6))
    plt.barh(range(10), python_bigram_frequencies, align='center')
    plt.yticks(range(10), top_10_python_bigrams)
    plt.xlabel('Frequency')
    plt.ylabel('Python Bigrams')
    plt.title('Top 10 Bigrams Occurring More Frequently in Python')
    plt.show()
    plt.figure(figsize=(10, 6))
    plt.barh(range(10), html_bigram_frequencies, align='center')
    plt.yticks(range(10), top_10_html_bigrams)
    plt.xlabel('Frequency')
    plt.ylabel('HTML Bigrams')
    plt.title('Top 10 Bigrams Occurring More Frequently in HTML')
    plt.show()

# =======================================================================================================
# visual4 END
# visual4 TO visual5
# visual5 START
# =======================================================================================================

def visual5(train):
    '''
    Shows the distribution of repositories that have at least one http word in them for both Python
    and HTML repositories as a ratio of total Python or HTML repositories

    INPUT:
    NONE

    OUTPUT:
    visual = Distribution of repositories with http words
    '''
    python_http_ratio = train[train.language == 'Python'].\
        cleaned_readme_contents.str.contains('srchttps_link').sum() / train[
        train.language == 'Python'].shape[0]
    html_http_ratio = train[train.language == 'HTML'].\
        cleaned_readme_contents.str.contains('srchttps_link').sum() / train[
        train.language == 'HTML'].shape[0]
    
    ratios = [python_http_ratio, html_http_ratio]
    labels = ['Python Repo w/ http Link', 'HTML Repo w/ http Link']
    
    plt.bar(labels, ratios)
    plt.ylabel('Ratio')
    plt.title('Ratio of http Links of Python Vs. HTML')
    plt.show()

# =======================================================================================================
# visual5 END
# visual5 TO stat1
# stat1 START
# =======================================================================================================

def stat1(train):
    '''
    Returns the statistical test to the visual5 function via the chi2_contingency statistical test
    by comparing the repository language and whether or not the repository had a srchttps_link
    within it

    INPUT:
    NONE

    OUTPUT:
    Stat = Accept/reject null hypothesis with the p-value
    '''
    alpha = 0.05
    observed = pd.crosstab(train.language,
                           train.cleaned_readme_contents.str.contains('srchttps_link'))
    chi2, p, _, hypothetical = stats.chi2_contingency(observed)
    
    if p < alpha:
        print('\033[32m========== REJECT THE NULL HYPOTHESIS! ==========\033[0m')
        print(f'\033[35mP-Value:\033[0m {p:.8f}')
        print(f'\033[35mChi-Squared-Value:\033[0m {chi2:.8f}')
    else:
        print('\033[31m========== ACCEPT THE NULL HYPOTHESIS! ==========\033[0m')
        print(f'\033[35mP-Value:\033[0m {p:.8f}')

# =======================================================================================================
# stat1 END
# stat1 TO baseline
# baseline START
# =======================================================================================================

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

# =======================================================================================================
# baseline END
# baseline TO models
# models START
# =======================================================================================================

def models():
    '''
    Returns a litany of models created for evaluation to acquire the best possible model for progression
    into the testing phase of model evaluations

    INPUT:
    NONE

    OUTPUT:
    metric_df = Elaboration of the top 15 models out of the 96 models created thus far
    '''
    repo_df = w.post_explore_wrangle_github_repositories()
    train, validate, test = w.train_split(repo_df)
    
    X_train = train.cleaned_readme_contents
    y_train = train.language
    X_validate = validate.cleaned_readme_contents
    y_validate = validate.language
    X_test = test.cleaned_readme_contents
    y_test = test.language
    
    train.language.value_counts(normalize=True)[0]
    accuracy = train.language.value_counts(normalize=True)[0]
    acc_validate = validate.language.value_counts(normalize=True)[0]
    metric_df = pd.DataFrame(data=[
        {
            'model' : 'baseline',
            'accuracy' : accuracy,
            'acc-validate' : acc_validate,
            'difference' : acc_validate - accuracy,
        }
    ])
    cv = CountVectorizer()
    X_bow = cv.fit_transform(X_train)
    tree = DecisionTreeClassifier(max_depth=3)
    tree.fit(X_bow, y_train)
    tfidf = TfidfVectorizer()
    bag_of_words_tfidf = tfidf.fit_transform(X_train)
    cv = CountVectorizer(ngram_range=(1, 3))
    bag_of_grams = cv.fit_transform(X_train)
    tfidf = CountVectorizer(ngram_range=(2, 3))
    bag_of_grams = tfidf.fit_transform(X_train)
    for i in range(1,4):
        for j in range(3,8):
            cv = CountVectorizer(ngram_range=(i, i))
            X_bow = cv.fit_transform(X_train)
            X_val_bow = cv.transform(X_validate)
            tree = DecisionTreeClassifier(max_depth=j)
            tree.fit(X_bow, y_train)
            accuracy = tree.score(X_bow, y_train)
            acc_validate = tree.score(X_val_bow, y_validate)
            
            metric_df = metric_df.append(
                {
                    'model' : f'decistion_tree-cv_{i}gram_{j}depth',
                    'accuracy' : accuracy,
                    'acc-validate' : acc_validate,
                    'difference' : acc_validate - accuracy,
                }, ignore_index=True
            )
    for i in range(1,4):
        for j in range(3,8):
            cv = CountVectorizer(ngram_range=(i, i))
            X_bow = cv.fit_transform(X_train)
            X_val_bow = cv.transform(X_validate)
            tree = RandomForestClassifier(max_depth=j)
            tree.fit(X_bow, y_train)
            accuracy = tree.score(X_bow, y_train)
            acc_validate = tree.score(X_val_bow, y_validate)
            
            metric_df = metric_df.append(
                {
                    'model' : f'random_forest-cv_{i}gram_{j}depth',
                    'accuracy' : accuracy,
                    'acc-validate' : acc_validate,
                    'difference' : acc_validate - accuracy,
                }, ignore_index=True
            )
    for i in range(1,4):
        for j in range(3,8):
            cv = CountVectorizer(ngram_range=(i, i))
            X_bow = cv.fit_transform(X_train)
            X_val_bow = cv.transform(X_validate)
            model = KNeighborsClassifier(n_neighbors=j)
            model.fit(X_bow, y_train)
            accuracy = model.score(X_bow, y_train)
            acc_validate = model.score(X_val_bow, y_validate)
            
            metric_df = metric_df.append(
                {
                    'model' : f'k_nearest-cv_{i}gram_{j}neighbors',
                    'accuracy' : accuracy,
                    'acc-validate' : acc_validate,
                    'difference' : acc_validate - accuracy,
                }, ignore_index=True
            )
    for i in range(1,4):
        cv = CountVectorizer(ngram_range=(i, i))
        X_bow = cv.fit_transform(X_train)
        X_val_bow = cv.transform(X_validate)
        model = LogisticRegression()
        model.fit(X_bow, y_train)
        accuracy = model.score(X_bow, y_train)
        acc_validate = model.score(X_val_bow, y_validate)

        metric_df = metric_df.append(
            {
                'model' : f'logistic_regress-cv_{i}gram',
                'accuracy' : accuracy,
                'acc-validate' : acc_validate,
                'difference' : acc_validate - accuracy,
            }, ignore_index=True
        )
    for i in range(1,4):
        for j in range(3,8):
            tfidf = TfidfVectorizer(ngram_range=(i, i))
            X_bow = tfidf.fit_transform(X_train)
            X_val_bow = tfidf.transform(X_validate)
            tree = DecisionTreeClassifier(max_depth=j)
            tree.fit(X_bow, y_train)
            accuracy = tree.score(X_bow, y_train)
            acc_validate = tree.score(X_val_bow, y_validate)
            
            metric_df = metric_df.append(
                {
                    'model' : f'decistion_tree-tfidf_{i}gram_{j}depth',
                    'accuracy' : accuracy,
                    'acc-validate' : acc_validate,
                    'difference' : acc_validate - accuracy,
                }, ignore_index=True
            )
    for i in range(1,4):
        for j in range(3,8):
            tfidf = TfidfVectorizer(ngram_range=(i, i))
            X_bow = tfidf.fit_transform(X_train)
            X_val_bow = tfidf.transform(X_validate)
            tree = RandomForestClassifier(max_depth=j)
            tree.fit(X_bow, y_train)
            accuracy = tree.score(X_bow, y_train)
            acc_validate = tree.score(X_val_bow, y_validate)
            
            metric_df = metric_df.append(
                {
                    'model' : f'random_forest-tfidf_{i}gram_{j}depth',
                    'accuracy' : accuracy,
                    'acc-validate' : acc_validate,
                    'difference' : acc_validate - accuracy,
                }, ignore_index=True
            )
    for i in range(1,4):
        for j in range(3,8):
            tfidf = TfidfVectorizer(ngram_range=(i, i))
            X_bow = tfidf.fit_transform(X_train)
            X_val_bow = tfidf.transform(X_validate)
            model = KNeighborsClassifier(n_neighbors=j)
            model.fit(X_bow, y_train)
            accuracy = model.score(X_bow, y_train)
            acc_validate = model.score(X_val_bow, y_validate)
            
            metric_df = metric_df.append(
                {
                    'model' : f'k_nearest-tfidf_{i}gram_{j}neighbors',
                    'accuracy' : accuracy,
                    'acc-validate' : acc_validate,
                    'difference' : acc_validate - accuracy,
                }, ignore_index=True
            )
    for i in range(1,4):
        tfidf = TfidfVectorizer(ngram_range=(i, i))
        X_bow = tfidf.fit_transform(X_train)
        X_val_bow = tfidf.transform(X_validate)
        model = LogisticRegression()
        model.fit(X_bow, y_train)
        accuracy = model.score(X_bow, y_train)
        acc_validate = model.score(X_val_bow, y_validate)

        metric_df = metric_df.append(
            {
                'model' : f'logistic_regress-tfidf_{i}gram',
                'accuracy' : accuracy,
                'acc-validate' : acc_validate,
                'difference' : acc_validate - accuracy,
            }, ignore_index=True
        )
    return metric_df.sort_values('acc-validate', ascending=False).head(15)

# =======================================================================================================
# models END
# models TO topmodel
# topmodel START
# =======================================================================================================

def topmodel():
    '''
    Demonstates the best model established from evaluating all models created and shows the scores
    during the testing phase of modeling

    INPUT:
    NONE

    OUTPUT:
    topmodel = Elaboration of the best model performance on the testing dataset (Unseen data)
    '''

# =======================================================================================================
# topmodel END
# =======================================================================================================