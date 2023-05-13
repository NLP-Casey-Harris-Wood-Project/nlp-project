# Predicting Python/HTML of Github NLP Repositories

<!-- ![Project Banner](path/to/banner_image.png) -->

*Project banner image credits: [Source Name](image_source_url)*

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Setup](#setup)
- [Data Preprocessing](#data-preprocessing)
- [Model Selection and Training](#model-selection-and-training)
- [Results](#results)
- [Future Work](#future-work)
- [Acknowledgements](#acknowledgements)

## Overview

<!-- - Project Description Describes what your project is and why it is important -->

<b><u><i>Project Description:</i></u></b>

Utilizing Web-Scraping techniques on Github NLP repository's README that have the majority of the code as Python or HTML, create a classification model that accurately predicts the predominant coding language used within each repository.  This is important to see if there is any pattern of vocabulary usage that tends to dictate the predominant coding language.

<!-- - Briefly describe the project goal, dataset, and problem statement. -->

<b><u><i>Project Goals:</i></u></b>

Identify patterns of vocabulary within README files to identify Python or HTML coding language from Github NLP repositories and create classification models in order to determine if there is a pattern of vocabulary usage unique to Python and unique to HTML.

<!-- - Project Goal Clearly states what your project sets out to do and how the information gained can be applied to the real world -->
<!-- - Initial Hypotheses Initial questions used to focus your project -->

<b><u><i>Initial Hypothesis:</i></u></b>

Since HTML and Python utilizes unique syntaxes and code, then there should be unique terminology used within README files that can identify the predominant coding language of a repository.

<b><u><i>Initial Questions:</i></u></b>

1. Is there unique terminology used for Python and HTML?
2. Are there words that are used more in Python than HTML and vice versa?
3. Are there 2-word combinations that are used more in Python than HTML and vice versa?
4. Are there 3-word combinations that are used more in Python than HTML and vice versa?

## Dataset

<!-- - Provide a description of the dataset, including the number of records, features, and target variable. -->

<b><u><i>Description:</i></u></b>

Web-scraped data from 500 NLP related Github repositories that contain the README contents and predominant code language (Python or HTML) for each repository.  The target variable is the predominant coding language of each repository (Python or HTML).

<!-- - Include a link to the dataset source, if available.
- Data dictionary -->

<b><u><i>Data Dictionary:</i></u></b>

| Feature Name | Data Type | Description | Example |
| ----- | ----- | ----- | ----- |
| repo_names | object | Name of Repository | 'huggingface/transformers' |
| contents | object | Contents of Repository's README file | 'Transformers provides thousands of pretrained...' |
| language | object | Predominant coding language of Repository | 'Python' |



## Setup

<!-- - Gives instructions for reproducing your work. i.e. Running your notebook on someone else's computer. -->

<b><u><i>Instructions to Reproduce:</i></u></b>

1. Clone this repository
2. Generate a Github Token

    - Go here: https://github.com/settings/tokens
    - Click: 'Generate New Token(Classic)'
    - DO NOT check any boxes
    - Copy TOKEN URL
3. Create 'env.py' file with:

    - github_username = YOUR GITHUB USERNAME
    - github_token = TOKEN URL
4. Run desired files

<!-- - List the required Python libraries and their versions. -->

| Python Library | Version | Usage |
| ----- | ----- | ----- |
| numpy | 1.21.5 | Vectorization |
| pandas | 1.4.4 | Dataframing |
| matplotlib | 3.5.2 | Visualization |
| seaborn | 0.11.2 | Visualization |
| wordcloud | 1.9.1.1 | Visualization |
| bs4 | 4.11.1 | NLP |
| requests | 2.28.1 | NLP |
| regex | 2022.7.9 | NLP |
| nltk | 3.7 | NLP |
| unicodedata | X | NLP |
| sklearn | 1.0.2 | Stats, Metrics, Modeling |



<!-- - Include instructions for setting up a virtual environment, if necessary.
- Provide any additional setup instructions, if needed. -->

## Data Preprocessing

<!-- - Project Plan Guides the reader through the different stages of the pipeline as they relate to your project
- Briefly describe the data preprocessing steps, including handling missing values, encoding categorical variables, scaling or normalizing numerical variables, and feature engineering. -->

<b><u><i>Missing Value Handling:<i></u></b>

Nothing of significance

<b><u><i>NLP Methodology:<i></u></b>

1. Clean text of contents
2. Tokenize cleaned text
3. Lemmatize tokenized data
4. Remove stop-words (To include predominant coding language) of lemmatized data

<b><u><i>Modeling Specific:<i></u></b>

1. Count Vectorizer (CV)

    - With or without ngram_range=(#, #)
2. Term Frequency - Inverse Document Frequency (TF-IDF)

    - With or without ngram_range=(#, #)

## Model Selection and Training

<!-- - List the machine learning models considered for the project.
- Explain the model selection process and criteria. -->

<b><u><i>Classification Models:</i></u></b>

- DecisionTreeClassifier()
- RandomForestClassifier()
- LogisticRegression()
- MultinomialNB()

<!-- - Describe the model training process, including hyperparameter tuning and cross-validation, if applicable. -->

<b><u><i>Training Procedure:</i></u></b>

1. Split into train, validate, and test sets
2. Using features selected, fit and transform on training data
3. Evaluate train and validate scores to determine best model
4. Run best model on the test set and evaluate results

<b><u><i>Model Evaluation Metric:</i></u></b>

- Accuracy
- Since we do not necessarily care for specifically Python or HTML predictions, but rather the overall accuracy of the model, we will evaluate models on their accuracy scores

<!-- ![Model Performance Comparison](path/to/model_performance_image.png) -->

*Image caption: A comparison of the performance of different models on the dataset.*

## Results

- Summarize the project results, including the best-performing model, its performance metrics, and any insights derived from the analysis.

## Future Work

- Discuss potential improvements, additional features, or alternative approaches for the project.

## Acknowledgements

- List any references, articles, or resources used during the project.
- Acknowledge any collaborators or external support, if applicable.

