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
12. post_explore_wrangle_github_repositories
13. train_split
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
from sklearn.model_selection import train_test_split

# ===============================================================================================
# Imports END
# Imports TO basic_clean
# basic_clean START
# ===============================================================================================

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
    # remove special characters
    removed_special = [re.sub(r'[^\w\s]', '', text) for text in entry]
    # convert text into unicode characters
    normalized = [unicodedata.normalize('NFKD',text).encode('ascii', 'ignore').\
                  decode('utf-8') for text in removed_special]
    # lowercase all the text
    return [text.lower() for text in normalized]
    
# ==============================================================================================
# basic_clean END
# basic_clean TO tokenize
# tokenize START
# ==============================================================================================

def tokenize(entry):
    '''
    Takes in a cleaned pandas series (column) and tokenizes all the words in the string
    
    INPUT:
    entry = Cleaned pandas series (Column) that needs to be tokenized
    
    OUTPUT:
    tokenized_data = Pandas series (Column) that is tokenized (I HAVE EXORCISED THE DEMON)
    '''
    # create a tokenizer object
    tokenizer = nltk.tokenize.toktok.ToktokTokenizer()
    # tokenize the data
    tokenized_data = [tokenizer.tokenize(text, return_str=True) for text in entry]
    # return the tokenized data
    return tokenized_data
    
# =============================================================================================
# tokenize END
# tokenize TO stem
# stem START
# ==============================================================================================

def stem(entry):
    '''
    Takes in a cleaned and tokenized pandas series (column) and applies stemming to all the words
    
    INPUT:
    entry = Cleaned and tokenized pandas series (column) that needs to be stemmed
    
    OUTPUT:
    stemmed_data = Pandas series (column) that is stemmed (I HAVE EXORCISED THE DEMON)
    '''
    # create a stemmer object
    stemmer = nltk.porter.PorterStemmer()
    # create empty list to store results
    stemmed_data = []
    # loop through the text
    for text in entry:
        # stem the tokenized words
        stemmed_tokens = [stemmer.stem(token) for token in text.split()]
        # join the list of words together into one string
        stemmed_text = ' '.join(stemmed_tokens)
        # add the string to the results list
        stemmed_data.append(stemmed_text)
    # return the list of strings
    return stemmed_data
    
# ===============================================================================================
# stem END
# stem TO lemmatize
# lemmatize START
# =============================================================================================

def lemmatize(entry):
    '''
    Takes in a cleaned and tokenized pandas series (column) and applies lemmatization to each word
    
    INPUT:
    entry = Cleaned and tokenized pandas series (column) that needs to be lemmatized
    
    OUTPUT:
    lemmatized_data = Pandas series (column) that is lemmatized (I HAVE EXORCISED THE DEMON)
    '''
    # create a lemmatizer object
    lemmatizer = nltk.stem.WordNetLemmatizer()
    # create an empty list to store the results
    lemmatized_data = []
    # loop through the text
    for text in entry:
        # lemmatize each word in the text
        lemmatized_tokens = [lemmatizer.lemmatize(word) for word in text.split()]
        # join together the split words into one string
        lemmatized_text = ' '.join(lemmatized_tokens)
        # add the string to the results list
        lemmatized_data.append(lemmatized_text)
    # return the list of strings
    return lemmatized_data
    
# ===============================================================================================
# lemmatize END
# lemmatize TO remove_stopwords
# remove_stopwords START
# ===============================================================================================

def remove_stopwords(entry, extra_removal_words=[], keep_words=[]):
    '''
    Takes in a cleaned, tokenized, and stemmed/lemmatized pandas series (column) 
    and removes all of the stopwords
    
    INPUT:
    entry = Cleaned, tokenized, and stemmed/lemmatized pandas series (column) 
    that needs stopwords removed
    extra_removal_words = Additional words to include for removal
    keep_words = Words to exclude from removal
    
    OUTPUT:
    removed_stopwords = Pandas series (column) that has stopwords removed 
    (I HAVE EXORCISED THE DEMON)
    '''
    # create a stopword object
    stopwords = nltk.corpus.stopwords
    # get a list of english stopwords
    stopwords_list = stopwords.words('english')
    # if there are extra words to add to the stopwords list, add them
    stopwords_list.extend(extra_removal_words)
    # if there are words from the stopwords lsit that we dont want remove, then remove them
    stopwords_list = list(set(stopwords_list) - set(keep_words))
    # create an empty list for the results
    removed_stopwords = []
    # loop through the text
    for text in entry:
        # if word is a stopword, then remove it
        not_stopword = [word for word in text.split() if word not in stopwords_list]
        # join the split words together into one string
        no_stopwords = ' '.join(not_stopword)
        # add the string to the results list
        removed_stopwords.append(no_stopwords)
    # return the list of strings
    return removed_stopwords

# =================================================================================================
# remove_stopwords END
# remove_stopwords TO full_clean
# full_clean START
# =================================================================================================

def full_clean(entry, stem_method=False, extra_removal_words=[], keep_words=[]):
    '''
    Takes in a pandas series (column) and conducts the basic_clean, tokenize, stem/lemmatize
    (Lemmatize by default), and the remove_stopwords functions all at once rather 
    than having to run the functions separately
    
    INPUT:
    entry = Pandas series (column) that needs to be cleaned from start to finish
    stem_method = Boolean True/False: False for stemming and True for lemmatizing
    extra_removal_words = Additional words to include for removal
    keep_words = Words to exclude from removal
    
    OUTPUT:
    full_cleaned_data = Pandas series (column) that has stopwords removed 
    (I HAVE EXORCISED THE DEMON)
    '''
    # convert text to lowercase unicode characters without special characters
    cleaned = basic_clean(entry)
    # tokenize the data
    tokenized = tokenize(cleaned)
    # if we are stemming the data
    if stem_method == True:
        # stem the data
        stemmed_or_lemmatized = stem(tokenized)
    # if we are lemmatizing the data
    else:
        # lemmatize the data
        stemmed_or_lemmatized = lemmatize(tokenized)
    # remove the stopwords
    removed_stopwords = remove_stopwords(stemmed_or_lemmatized,
                                         extra_removal_words=extra_removal_words,
                                         keep_words=keep_words)
    # rename the variable
    full_cleaned_data = removed_stopwords
    # return the cleaned data
    return full_cleaned_data

# ================================================================================================
# full_clean END
# full_clean TO acquire_github_repositories
# acquire_github_repositories START
# =================================================================================================

def acquire_github_repositories():
    '''
    Acquires the vanilla github repository data.
    
    INPUT:
    NONE
    
    OUTPUT:
    acquire_github_df = Vanilla pandas dataframe of github repositories
    '''
    # scrape the readme data from our repo list
    raw_data = a.scrape_github_data()
    # convert our data to a dataframe
    acquire_github_df = pd.DataFrame(raw_data)
    # return the dataframe of our data
    return acquire_github_df

# =================================================================================================
# acquire_github_repositories END
# acquire_github_repositories TO prepare_github_repositories
# prepare_github_repositories START
# ================================================================================================

def prepare_github_repositories():
    '''
    Takes in the vanilla github repository and prepares the data for exploration 
    and modeling purposes.
    Drops rows if contents na, normalizes contents, and removes any instances of 
    language in contents
    First checks to see if 'repo.csv' exists prior to running anything else.
    
    INPUT:
    NONE
    
    OUTPUT:
    prepared_github_df = Prepared pandas dataframe of github repositories
    '''
    # check if a cached version of the dataset exists in the local directory
    if os.path.exists('repo.csv'):
        # read in cached data
        vanilla_df = pd.read_csv('repo.csv', index_col=0)
    # if there is not a cached version of the data locally
    else:
        # scrape the readme data from github
        vanilla_df = acquire_github_repositories()
        
    # check if there are null values in the readme data
    if vanilla_df.readme_contents.isna().sum() > 1:
        # remove null readmes
        vanilla_df = vanilla_df[vanilla_df.readme_contents.isna() == False]
        # reset the index
        vanilla_df.reset_index(inplace=True)
        # drop the index column
        vanilla_df.drop(columns='index', inplace=True)

    # create dataframes with the data split into html and python readmes
    html_only_df = vanilla_df[vanilla_df.language == 'HTML']
    python_only_df = vanilla_df[vanilla_df.language == 'Python']

    # remove 'html' from html readmes, and 'python' from python readmes
    # using the stopwords removal function
    python_only_df['cleaned_readme_contents'] = full_clean(python_only_df.readme_contents,
                                                           extra_removal_words=['python'])
    html_only_df['cleaned_readme_contents'] = full_clean(html_only_df.readme_contents,
                                                         extra_removal_words=['html'])
    # create an empty list for the python results
    removed_target_python = []
    # remove 'python' from the python readmes using regex
    for row in python_only_df.cleaned_readme_contents:
        removed_target_python.append(re.sub(r'\w?python\w?', '', row))
    # replace the python_df readmes with the 'python' removed text
    python_only_df.cleaned_readme_contents = removed_target_python
    # create an empty list for the html results
    removed_target_html = []
    # remove 'html' from the html readmes using regex
    for row in html_only_df.cleaned_readme_contents:
        removed_target_html.append(re.sub(r'\w?html\w?', '', row))
    # replace the html_df readmes with the 'html' removed text
    html_only_df.cleaned_readme_contents = removed_target_html
    # combine the python only df and and html only df
    prepared_github_df = pd.concat([python_only_df, html_only_df], axis=0)
    # return the cleaned df
    return prepared_github_df

# ==============================================================================================
# prepare_github_repositories END
# prepare_github_repositories TO wrangle_github_repositories
# wrangle_github_repositories START
# ===============================================================================================

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
    # check if there is a cached version of the dataset locally
    if os.path.exists('repo.csv'):
        # read in the cached data
        wrangled_github_df = pd.read_csv('repo.csv', index_col=0)
        # return the cached data
        return wrangled_github_df
    # if there is no local cache of data
    else:
        # call the data scraping function
        wrangle_github_df = prepare_github_repositories()
        # write the scraped and prepared data to a csv file
        wrangle_github_df.to_csv('repo.csv')
        # return the data
        return wrangle_github_df

# ===============================================================================================
# wrangle_github_repositories END
# wrangle_github_repositories TO post_explore_wrangle_github_repositories
# post_explore_wrangle_github_repositories
# ================================================================================================

def post_explore_wrangle_github_repositories():
    '''
    From the wrangled 'repo.csv' data, additional preparation is applied in lieu of findings during
    the exploratory phase and returns a new dataframe that reflects the 
    changes made during exploration

    INPUT:
    NONE

    OUTPUT:
    post_explore_version = Pandas dataframe of findings from explore phase
    '''
    # acquire and prepare data according to the pre-exploration steps
    old_github_df = wrangle_github_repositories()
    # remove the outlier repo found during exploration
    old_github_df = old_github_df[~old_github_df.repo.str.startswith('pemistahl')]
    
    # create an empty list for results
    srchttps_text = []
    # loop through the readmes
    for text in old_github_df.cleaned_readme_contents.astype(str):
        # remove hyperlinks begining with 'srchttps'
        regexp = r'srchttps\w+'
        # add the new readme text to the results list
        srchttps_text.append(re.sub(regexp, 'srchttps_link', text))
    # replace the readmes in the df with the results text
    old_github_df.cleaned_readme_contents = srchttps_text
    
    # create an empty list for results
    weird_text = []
    # loop through each readme in the dataset
    for text in old_github_df.cleaned_readme_contents.astype(str):
        # remove the text '&#9' that isn't helping our results
        regexp = r'&#9;'
        # add the new readme text ot the results list
        weird_text.append(re.sub(regexp, '', text))
    # replace the readmes in the df with the results text
    old_github_df.cleaned_readme_contents = weird_text
    # create a new df with the df results
    post_explore_version = old_github_df
    # return the new df
    return post_explore_version

# ================================================================================================
# post_explore_wrangle_github_repositories END
# post_explore_wrangle_github_repositories TO split
# train_split START
# ===============================================================================================

def train_split(df):
    '''
    Takes in a dataframe and splits the data into train, validate, and test sets 
    with 70%, 20%, 10% of data.

    INPUT:
    df = Pandas dataframe to be splitted

    OUTPUT:
    train = Pandas dataframe with 70% of original data
    validate = Pandas dataframe with 20% of original data
    test = Pandas dataframe with 10% of original data
    '''
    # split off the test dataset
    train_validate, test = train_test_split(df,
                                            # set the random seed
                                            random_state=1349,
                                            # set the size of the test dataset
                                            train_size=0.9,
                                            # get an even distribution of language in each dataset
                                            stratify=df.language)
    # split the train and validation datasets
    train, validate = train_test_split(train_validate,
                                       # set the random seed
                                       random_state=1349,
                                       # set the size of the train and validate datasets
                                       train_size=0.778,
                                       # get an even distribution of language in each dataset
                                       stratify=train_validate.language)
    # return the split datasets
    return train, validate, test

# =============================================================================================
# train_split END
# =============================================================================================