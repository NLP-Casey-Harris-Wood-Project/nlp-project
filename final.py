# ===============================================================================================
# Table of Contents START
# ===============================================================================================

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

# ================================================================================================
# Table of Contents END
# Table of Contents TO Orientation
# Orientation START
# ================================================================================================

'''
The purpose of this file is to create functions specifically for the 'final_report.ipynb'
file and anything else that may be of use.
'''

# ================================================================================================
# Orientation END
# Orientation TO Imports
# Imports START
# ================================================================================================

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

# ===============================================================================================
# Imports END
# Imports TO vanilla
# vanilla START
# ===============================================================================================

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

# ================================================================================================
# vanilla END
# vanilla TO prepare
# prepare START
# ================================================================================================

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

# ===============================================================================================
# prepare END
# prepare TO train_split
# train_split START
# ===============================================================================================

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

# ===============================================================================================
# train_split END
# train_split TO get_words
# get_words START
# ===============================================================================================

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

# ==============================================================================================
# get_words END
# get_words TO visual1
# visual1 START
# ==============================================================================================

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
    # get the python repos from the df before remove the outlier repo
    python_df = before_df[before_df['language'] == 'Python']
    # get repos that contain hyperlinks 'srchttps'
    matched_repos = python_df[python_df['cleaned_readme_contents'].str.contains("srchttps")]
    # get count of how many times hyperlinks appear per repo
    matched_repos['frequency'] = matched_repos['cleaned_readme_contents'].str.count("srchttps")
    # sort by frequency of hyperlinks
    sorted_repos = matched_repos.sort_values('frequency', ascending=False)
    # convert the repo list and http frequency to a list
    repo_freq_list = sorted_repos[['repo', 'frequency']].values.tolist()
    # get a list of the repos
    repos = [repo for repo, freq in repo_freq_list]
    # get a list of the http frequencys
    frequencies = [freq for repo, freq in repo_freq_list]
    
    # get the python repos from the df after removing the outlier repo
    python_df_after = after_df[after_df['language'] == 'Python']
    # get repos that contain hyperlinks 'scrhttps
    matched_repos_after = python_df_after[python_df_after['cleaned_readme_contents'].\
                                          str.contains("srchttps")]
    # get count of how many times hyperlinks appear per repo
    matched_repos_after['frequency'] = matched_repos_after['cleaned_readme_contents'].\
                                        str.count("srchttps")
    # sort by frequency of hyperlinks
    sorted_repos_after = matched_repos_after.sort_values('frequency', ascending=False)
    # convert the repo list and http frequency to a list
    repo_freq_list_after = sorted_repos_after[['repo', 'frequency']].values.tolist()
    # get a list of the repos
    repos_after = [repo for repo, freq in repo_freq_list_after]
    # get a list of the http frequencys
    frequencies_after = [freq for repo, freq in repo_freq_list_after]
    
    # set the visualization styling
    plt.style.use('ggplot')
    # create subplots
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    # create horizontal bar graph of http frequency by repo before removing outlier repo
    axs[0].barh(repos, frequencies)
    # add axis labels
    axs[0].set_xlabel('Frequency')
    axs[0].set_ylabel('Repository')
    # add title
    axs[0].set_title('Frequency of "srchttps" in Repositories (Before)')
    # remove the y ticks since they are unreadable
    axs[0].set_yticks([])
    # create horizontal bar graph of http frequency by repo after removing outlier repo
    axs[1].barh(repos_after, frequencies_after)
    # add axis labels
    axs[1].set_xlabel('Frequency')
    axs[1].set_ylabel('Repository')
    # add title
    axs[1].set_title('Frequency of "srchttps" in Repositories (After)')
    # remove the y ticks since they are unreadable
    axs[1].set_yticks([])
    # move the subplots together
    plt.tight_layout()
    # display the plots
    plt.show()

# ===============================================================================================
# visual1 END
# visual1 TO visual2
# visual2 START
# ===============================================================================================

def visual2(train, python_text, html_text):
    '''
    Gets the distribution of unique words for both Python and HTML repositories

    INPUT:
    NONE

    OUTPUT:
    visual = Subplot with 2 distribution visuals of unique words, one for Python, one for HTML
    '''
    # get tokenized words in python repos
    python_tokens = word_tokenize(python_text)
    # get the frequency words appear in python repos
    python_freqdist = FreqDist(python_tokens)
    
    # get tokenized words in html repos
    html_tokens = word_tokenize(html_text)
    # get the frequency words appear in html repos
    html_freqdist = FreqDist(html_tokens)
    
    # get tuples of each word in python repos and the frequency it appears 
    # only if the word doesn't appear in html repos
    python_word_freq = [(word, freq) for word, freq in python_freqdist.items() 
                        if word not in html_freqdist.keys()]
    # sort the word list
    sorted_python_word_freq = sorted(python_word_freq, key=lambda x: x[1], reverse=True)
    # get the top 10 most frequent words used in python repos
    top_10_python_words = [pair[0] for pair in sorted_python_word_freq][:10]
    # get the frequency of the top 10 most frequent words in python repos
    top_10_python_frequencies = [pair[1] for pair in sorted_python_word_freq][:10]
    
    # get tuples of each word in html repos and the frequency it appears 
    # only if the word doesn't appear in python repos
    html_word_freq = [(word, freq) for word, freq in 
                      html_freqdist.items() if word not in python_freqdist.keys()]
    # sort the word list
    sorted_html_word_freq = sorted(html_word_freq, key=lambda x: x[1], reverse=True)
    # get the top 10 most frequent words used in python repos
    top_10_html_words = [pair[0] for pair in sorted_html_word_freq][:10]
    # get the frequency of the top 10 most frequent words in python repos
    top_10_html_frequencies = [pair[1] for pair in sorted_html_word_freq][:10]
    
    # create a dictionary of results
    data = {
        'Python Word': top_10_python_words,
        'Python Frequency': top_10_python_frequencies,
        'HTML Word': top_10_html_words,
        'HTML Frequency': top_10_html_frequencies
    }
    # convert results to dataframe
    df = pd.DataFrame(data)
    
    # get the python words
    df_python_sorted = df.sort_values(by='Python Frequency', ascending=False)
    # get the html words
    df_html_sorted = df.sort_values(by='HTML Frequency', ascending=False)
    
    # set the visualization styling
    plt.style.use('ggplot')
    # create subplots
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


    
# ===============================================================================================
# visual2 END
# visual2 TO visual3
# visual3 START
# ===============================================================================================

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

# ===============================================================================================
# visual3 END
# visual3 TO visual4
# visual4 START
# ===============================================================================================

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

# ===============================================================================================
# visual4 END
# visual4 TO visual5
# visual5 START
# ===============================================================================================

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

# ===============================================================================================
# visual5 END
# visual5 TO stat1
# stat1 START
# ===============================================================================================

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

# ==============================================================================================
# stat1 END
# ===============================================================================================