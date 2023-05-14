# =======================================================================================================
# Table of Contents START
# =======================================================================================================

'''
1. Orientation
2. Imports
3. basic_clean
4. tokenize
5. stem
6. lemmatize
7. remove_stopwords
8. full_clean
9. acquire_github_repositories
10. prepare_github_repositories
11. wrangle_github_repositories
'''

# =======================================================================================================
# Table of Contents END
# Table of Contents TO Orientation
# Orientation START
# =======================================================================================================

'''
The purpose of this file is to create functions specifically for the 'prepare.ipynb'
file and anything else that may be of use.
'''

# =======================================================================================================
# Orientation END
# Orientation TO Imports
# Imports START
# =======================================================================================================

import numpy as np
import pandas as pd
import re
import nltk
import unicodedata
import acquire as a
import os

# =======================================================================================================
# Imports END
# Imports TO basic_clean
# basic_clean START
# =======================================================================================================

def basic_clean(entry):
    '''
    Takes in a pandas series (column), lowercases everything, normalizes unicode characters,
    and replaces anything that is not a letter, number, whitespace or a
    single quote
    
    INPUT:
    entry = Pandas series (column) that needs to be cleaned
    
    OUTPUT:
    cleaned = Pandas series (column) that is cleaned (I HAVE EXORCISED THE DEMON)
    '''
    removed_special = [re.sub(r'[^\w\s]', '', text) for text in entry]
    normalized = [unicodedata.normalize('NFKD',text).encode('ascii', 'ignore').decode('utf-8') for text in removed_special]
    lowered = [text.lower() for text in normalized]
    cleaned = lowered
    return cleaned
    
# =======================================================================================================
# basic_clean END
# basic_clean TO tokenize
# tokenize START
# =======================================================================================================

def tokenize(entry):
    '''
    Takes in a cleaned pandas series (column) and tokenizes all the words in the string
    
    INPUT:
    entry = Cleaned pandas series (Column) that needs to be tokenized
    
    OUTPUT:
    tokenized_data = Pandas series (Column) that is tokenized (I HAVE EXORCISED THE DEMON)
    '''
    tokenizer = nltk.tokenize.toktok.ToktokTokenizer()
    tokenized_data = [tokenizer.tokenize(text, return_str=True) for text in entry]
    return tokenized_data
    
# =======================================================================================================
# tokenize END
# tokenize TO stem
# stem START
# =======================================================================================================

def stem(entry):
    '''
    Takes in a cleaned and tokenized pandas series (column) and applies stemming to all the words
    
    INPUT:
    entry = Cleaned and tokenized pandas series (column) that needs to be stemmed
    
    OUTPUT:
    stemmed_data = Pandas series (column) that is stemmed (I HAVE EXORCISED THE DEMON)
    '''
    stemmer = nltk.porter.PorterStemmer()
    stemmed_data = []
    for text in entry:
        stemmed_tokens = [stemmer.stem(token) for token in text.split()]
        stemmed_text = ' '.join(stemmed_tokens)
        stemmed_data.append(stemmed_text)
    return stemmed_data
    
# =======================================================================================================
# stem END
# stem TO lemmatize
# lemmatize START
# =======================================================================================================

def lemmatize(entry):
    '''
    Takes in a cleaned and tokenized pandas series (column) and applies lemmatization to each word
    
    INPUT:
    entry = Cleaned and tokenized pandas series (column) that needs to be lemmatized
    
    OUTPUT:
    lemmatized_data = Pandas series (column) that is lemmatized (I HAVE EXORCISED THE DEMON)
    '''
    lemmatizer = nltk.stem.WordNetLemmatizer()
    lemmatized_data = []
    for text in entry:
        lemmatized_tokens = [lemmatizer.lemmatize(word) for word in text.split()]
        lemmatized_text = ' '.join(lemmatized_tokens)
        lemmatized_data.append(lemmatized_text)
    return lemmatized_data
    
# =======================================================================================================
# lemmatize END
# lemmatize TO remove_stopwords
# remove_stopwords START
# =======================================================================================================

def remove_stopwords(entry, extra_removal_words=[], keep_words=[]):
    '''
    Takes in a cleaned, tokenized, and stemmed/lemmatized pandas series (column) and removes all of the stopwords
    
    INPUT:
    entry = Cleaned, tokenized, and stemmed/lemmatized pandas series (column) that needs stopwords removed
    extra_removal_words = Additional words to include for removal
    keep_words = Words to exclude from removal
    
    OUTPUT:
    removed_stopwords = Pandas series (column) that has stopwords removed (I HAVE EXORCISED THE DEMON)
    '''
    stopwords = nltk.corpus.stopwords
    stopwords_list = stopwords.words('english')
    stopwords_list.extend(extra_removal_words)
    stopwords_list = list(set(stopwords_list) - set(keep_words))
    removed_stopwords = []
    for text in entry:
        not_stopword = [word for word in text.split() if word not in stopwords_list]
        no_stopwords = ' '.join(not_stopword)
        removed_stopwords.append(no_stopwords)
    return removed_stopwords

# =======================================================================================================
# remove_stopwords END
# remove_stopwords TO full_clean
# full_clean START
# =======================================================================================================

def full_clean(entry, stem_method=False, extra_removal_words=[], keep_words=[]):
    '''
    Takes in a pandas series (column) and conducts the basic_clean, tokenize, stem/lemmatize
    (Lemmatize by default), and the remove_stopwords functions all at once rather than having to run the functions separately
    
    INPUT:
    entry = Pandas series (column) that needs to be cleaned from start to finish
    stem_method = Boolean True/False: False for stemming and True for lemmatizing
    extra_removal_words = Additional words to include for removal
    keep_words = Words to exclude from removal
    
    OUTPUT:
    full_cleaned_data = Pandas series (column) that has stopwords removed (I HAVE EXORCISED THE DEMON)
    '''
    cleaned = basic_clean(entry)
    tokenized = tokenize(cleaned)
    if stem_method == True:
        stemmed_or_lemmatized = stem(tokenized)
    else:
        stemmed_or_lemmatized = lemmatize(tokenized)
    removed_stopwords = remove_stopwords(stemmed_or_lemmatized, extra_removal_words=extra_removal_words, keep_words=keep_words)
    full_cleaned_data = removed_stopwords
    return full_cleaned_data

# =======================================================================================================
# full_clean END
# full_clean TO acquire_github_repositories
# acquire_github_repositories START
# =======================================================================================================

def acquire_github_repositories():
    '''
    Acquires the vanilla github repository data.
    
    INPUT:
    NONE
    
    OUTPUT:
    acquire_github_df = Vanilla pandas dataframe of github repositories
    '''
    raw_data = a.scrape_github_data()
    acquire_github_df = pd.DataFrame(raw_data)
    return acquire_github_df

# =======================================================================================================
# acquire_github_repositories END
# acquire_github_repositories TO prepare_github_repositories
# prepare_github_repositories START
# =======================================================================================================

def prepare_github_repositories():
    '''
    Takes in the vanilla github repository and prepares the data for exploration and modeling purposes.
    Drops rows if contents na, normalizes contents, and removes any instances of language in contents
    First checks to see if 'repo.csv' exists prior to running anything else.
    
    INPUT:
    NONE
    
    OUTPUT:
    prepared_github_df = Prepared pandas dataframe of github repositories
    '''
    if os.path.exists('repo.csv'):
        vanilla_df = pd.read_csv('repo.csv', index_col=0)
        if vanilla_df.readme_contents.isna().sum() > 1:
            vanilla_df = vanilla_df[vanilla_df.readme_contents.isna() == False]
            vanilla_df.reset_index(inplace=True)
            vanilla_df.drop(columns='index', inplace=True)
        html_only_df = vanilla_df[vanilla_df.language == 'HTML']
        python_only_df = vanilla_df[vanilla_df.language == 'Python']
        python_only_df['cleaned_readme_contents'] = full_clean(python_only_df.readme_contents, extra_removal_words=['python'])
        html_only_df['cleaned_readme_contents'] = full_clean(html_only_df.readme_contents, extra_removal_words=['html'])
        removed_target_python = []
        for row in python_only_df.cleaned_readme_contents:
            removed_target_python.append(re.sub(r'\w?python\w?', '', row))
        python_only_df.cleaned_readme_contents = removed_target_python
        removed_target_html = []
        for row in html_only_df.cleaned_readme_contents:
            removed_target_html.append(re.sub(r'\w?html\w?', '', row))
        html_only_df.cleaned_readme_contents = removed_target_html
        prepared_github_df = pd.concat([python_only_df, html_only_df], axis=0)
        return prepared_github_df
    else:
        vanilla_df = acquire_github_repositories()
        if vanilla_df.readme_contents.isna().sum() > 1:
            vanilla_df = vanilla_df[vanilla_df.readme_contents.isna() == False]
            vanilla_df.reset_index(inplace=True)
            vanilla_df.drop(columns='index', inplace=True)
        html_only_df = vanilla_df[vanilla_df.language == 'HTML']
        python_only_df = vanilla_df[vanilla_df.language == 'Python']
        python_only_df['cleaned_readme_contents'] = full_clean(python_only_df.readme_contents, extra_removal_words=['python'])
        html_only_df['cleaned_readme_contents'] = full_clean(html_only_df.readme_contents, extra_removal_words=['html'])
        removed_target_python = []
        for row in python_only_df.cleaned_readme_contents:
            removed_target_python.append(re.sub(r'\w?python\w?', '', row))
        python_only_df.cleaned_readme_contents = removed_target_python
        removed_target_html = []
        for row in html_only_df.cleaned_readme_contents:
            removed_target_html.append(re.sub(r'\w?html\w?', '', row))
        html_only_df.cleaned_readme_contents = removed_target_html
        prepared_github_df = pd.concat([python_only_df, html_only_df], axis=0)
        return prepared_github_df

# =======================================================================================================
# prepare_github_repositories END
# prepare_github_repositories TO wrangle_github_repositories
# wrangle_github_repositories START
# =======================================================================================================

def wrangle_github_repositories():
    '''
    Acquires and prepares the github repository for exploration and modeling purposes.
    Drops rows if contents is na, normalizes contents, and removes any instances of language in contents.
    First checks to see if 'repo.csv' exists prior to running anything else.
    
    INPUT:
    NONE
    
    OUTPUT:
    repo.csv = .csv file IF NON-EXISTANT
    wrangled_github_df = Prepared pandas dataframe of github repositories
    '''
    if os.path.exists('repo.csv'):
        wrangled_github_df = pd.read_csv('repo.csv', index_col=0)
        return wrangled_github_df
    else:
        wrangle_github_df = prepare_github_repositories()
        wrangle_github_df.to_csv('repo.csv')
        return wrangle_github_df

# =======================================================================================================
# wrangle_github_repositories END
# =======================================================================================================