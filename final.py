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
import re
import unicodedata
import wrangle as w
import os
from sklearn.model_selection import train_test_split

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
    vanilla = pd.read_csv('repo.csv')
    return vanilla.shape

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
    repo_df = w.post_explore_wrangle_github_repositories()
    return repo_df

# =======================================================================================================
# prepare END
# prepare TO visual1
# visual1 START
# =======================================================================================================

def visual1():
    '''
    Shows the distribution of 'srchttps' per repository as a subplot with 2 visuals that demonstrate
    the distribution before the outlier removal and one after the outlier removal

    INPUT:
    NONE

    OUTPUT:
    visual = Subplot with 2 distribution plots, one before outlier removal, one after outlier removal
    '''
    # Assuming you have a DataFrame named 'repos_df' that contains repository information
    before_df = w.wrangle_github_repositories()
    python_df = before_df[before_df['language'] == 'Python']
    matched_repos = python_df[python_df['cleaned_readme_contents'].str.contains("srchttps")]
    matched_repos['frequency'] = matched_repos['cleaned_readme_contents'].str.count("srchttps")
    sorted_repos = matched_repos.sort_values('frequency', ascending=False)
    repo_freq_list = sorted_repos[['repo', 'frequency']].values.tolist()
    repos = [repo for repo, freq in repo_freq_list]
    frequencies = [freq for repo, freq in repo_freq_list]
    after_df = w.post_explore_wrangle_github_repositories()
    python_df_after = after_df[after_df['language'] == 'Python']
    matched_repos_after = python_df_after[python_df_after['cleaned_readme_contents'].str.contains("srchttps")]
    matched_repos_after['frequency'] = matched_repos_after['cleaned_readme_contents'].str.count("srchttps")
    sorted_repos_after = matched_repos_after.sort_values('frequency', ascending=False)
    repo_freq_list_after = sorted_repos_after[['repo', 'frequency']].values.tolist()
    repos_after = [repo for repo, freq in repo_freq_list_after]
    frequencies_after = [freq for repo, freq in repo_freq_list_after]
    plt.style.use('ggplot')
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

def visual2():
    '''
    Gets the distribution of unique words for both Python and HTML repositories

    INPUT:
    NONE

    OUTPUT:
    visual = Subplot with 2 distribution visuals of unique words, one for Python, one for HTML
    '''
    repo_df = w.post_explore_wrangle_github_repositories()
    train, validate, test = w.train_split(repo_df)
    python_df = train[train['language'] == 'Python']
    html_df = train[train['language'] == 'HTML']
    python_text = ' '.join(python_df['cleaned_readme_contents'])
    python_text = python_text.lower()
    python_tokens = word_tokenize(python_text)
    python_freqdist = FreqDist(python_tokens)
    html_text = ' '.join(html_df['cleaned_readme_contents'])
    html_text = html_text.lower()
    html_tokens = word_tokenize(html_text)
    html_freqdist = FreqDist(html_tokens)
    python_word_freq = [(word, freq) for word, freq in python_freqdist.items() if word not in html_freqdist.keys()]
    sorted_python_word_freq = sorted(python_word_freq, key=lambda x: x[1], reverse=True)
    top_10_python_words = [pair[0] for pair in sorted_python_word_freq][:10]
    top_10_python_frequencies = [pair[1] for pair in sorted_python_word_freq][:10]
    html_word_freq = [(word, freq) for word, freq in html_freqdist.items() if word not in python_freqdist.keys()]
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
    axes[0].barh(df_python_sorted['Python Word'], df_python_sorted['Python Frequency'], color='tab:blue')
    axes[0].set_title('Top 10 Unique Words: Python')
    axes[0].set_xlabel('Frequency')
    axes[0].set_ylabel('Words')
    axes[1].barh(df_html_sorted['HTML Word'], df_html_sorted['HTML Frequency'], color='tab:orange')
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

def visual3():
    '''
    Creates wordclouds of most commonly used words across repositories for both Python and HTML

    INPUT:
    NONE

    OUTPUT:
    visual = Subplot with 2 wordclouds, one for Python, one for HTML
    '''
    repo_df = w.post_explore_wrangle_github_repositories()
    train, validate, test = w.train_split(repo_df)
    python_words = ' '.join(train[train.language == 'Python']['cleaned_readme_contents'].astype(str))
    html_words = ' '.join(train[train.language == 'HTML']['cleaned_readme_contents'].astype(str))
    unique_python_words = set(python_words.split())
    unigram_python_img = WordCloud(background_color='white').generate(' '.join(unique_python_words))
    unique_html_words = set(html_words.split())
    unigram_html_img = WordCloud(background_color='white').generate(' '.join(unique_html_words))
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(unigram_python_img)
    axs[0].axis('off')
    axs[0].set_title('Most Common Python Words')
    axs[1].imshow(unigram_html_img)
    axs[1].axis('off')
    axs[1].set_title('Most Common HTML Words')
    plt.tight_layout()
    plt.show()

# =======================================================================================================
# visual3 END
# visual3 TO visual4
# visual4 START
# =======================================================================================================

def visual4():
    '''
    Returns a subplot of the distribution of bigrams for both Python and HTML repositories

    INPUT:
    NONE

    OUTPUT:
    visual = Subplot with 2 distribution visuals, one for Python, one for HTML
    '''
    repo_df = w.post_explore_wrangle_github_repositories()
    train, validate, test = w.train_split(repo_df)
    python_df = train[train['language'] == 'Python']
    html_df = train[train['language'] == 'HTML']
    python_text = ' '.join(python_df['cleaned_readme_contents'])
    html_text = ' '.join(html_df['cleaned_readme_contents'])
    python_bigrams = list(nltk.bigrams(nltk.word_tokenize(python_text)))
    html_bigrams = list(nltk.bigrams(nltk.word_tokenize(html_text)))
    python_bigram_freqdist = FreqDist(python_bigrams)
    html_bigram_freqdist = FreqDist(html_bigrams)
    python_unique_bigrams = set(python_bigram_freqdist.keys()) - set(html_bigram_freqdist.keys())
    html_unique_bigrams = set(html_bigram_freqdist.keys()) - set(python_bigram_freqdist.keys())
    sorted_python_bigrams = sorted(python_unique_bigrams, key=lambda x: python_bigram_freqdist[x], reverse=True)
    sorted_html_bigrams = sorted(html_unique_bigrams, key=lambda x: html_bigram_freqdist[x], reverse=True)
    top_10_python_bigrams = sorted_python_bigrams[:10]
    python_bigram_frequencies = [python_bigram_freqdist[bigram] for bigram in top_10_python_bigrams]
    top_10_html_bigrams = sorted_html_bigrams[:10]
    html_bigram_frequencies = [html_bigram_freqdist[bigram] for bigram in top_10_html_bigrams]
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

def visual5():
    '''
    Shows the distribution of repositories that have at least one http word in them for both Python
    and HTML repositories as a ratio of total Python or HTML repositories

    INPUT:
    NONE

    OUTPUT:
    visual = Distribution of repositories with http words
    '''
    repo_df = w.post_explore_wrangle_github_repositories()
    train, validate, test = w.train_split(repo_df)
    python_http_ratio = train[train.language == 'Python'].cleaned_readme_contents.str.contains('srchttps_link').sum() / train[train.language == 'Python'].shape[0]
    html_http_ratio = train[train.language == 'HTML'].cleaned_readme_contents.str.contains('srchttps_link').sum() / train[train.language == 'HTML'].shape[0]
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

def stat1():
    '''
    Returns the statistical test to the visual5 function via the chi2_contingency statistical test
    by comparing the repository language and whether or not the repository had a srchttps_link
    within it

    INPUT:
    NONE

    OUTPUT:
    Stat = Accept/reject null hypothesis with the p-value
    '''
    repo_df = w.post_explore_wrangle_github_repositories()
    train, validate, test = w.train_split(repo_df)
    alpha = 0.05
    observed = pd.crosstab(train.language, train.cleaned_readme_contents.str.contains('srchttps_link'))
    p = stats.chi2_contingency(observed)[1]
    if p < alpha:
        print('\033[32m========== REJECT THE NULL HYPOTHESIS! ==========\033[0m')
        print(f'\033[35mP-Value:\033[0m {p:.8f}')
    else:
        print('\033[31m========== ACCEPT THE NULL HYPOTHESIS! ==========\033[0m')
        print(f'\033[35mP-Value:\033[0m {p:.8f}')

# =======================================================================================================
# stat1 END
# stat1 TO baseline
# baseline START
# =======================================================================================================



# =======================================================================================================
# baseline END
# baseline TO models
# models START
# =======================================================================================================



# =======================================================================================================
# models END
# models TO topmodel
# topmodel START
# =======================================================================================================



# =======================================================================================================
# topmodel END
# =======================================================================================================