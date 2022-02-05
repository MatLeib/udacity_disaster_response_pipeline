# Importing libraries
from workspace_utils import active_session

import sys

import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger', 'stopwords'])

import re
import pandas as pd
from sqlalchemy import create_engine
import pickle

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'


def load_data(database_filepath):
    """Loading data from SQL database.
    
    Args:
        database_filepath (str): filepath of SQL database
    Returns:
        X (array): input variables X for machine learning model
        y (array): output variables y for machine leanring model
        category_names (list): label names
    """
    # Creating SQL engine
    engine = create_engine('sqlite:///'+str(database_filepath))
    
    # Reading data from SQL table
    df = pd.read_sql_table('Disasters', engine)
    
    # Selecting input variables, output variables and label names
    X = df.message.values
    y = df.iloc[:,4:].values
    category_names = df.iloc[:,4:].columns.tolist()
    
    return  X, y, category_names


def tokenize(text):
    """Preparing text data for machine learning model, including:
    - detecting urls and replacing them with place holder
    - tokenizing and lemmatizing
    - removing stop words
    - removing non-alphanumeric entries
    
    Args:
        text (str): text that should be prepared
    Returns:
        clean_tokens (list): list of cleaned tokens/words
    """
    # Cleaning urls
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
    
    # Tokenizing and Lemmatizing
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(lemmatizer.lemmatize(tok), pos='v').lower().strip()
        clean_tokens.append(clean_tok)
    
    # Removing stop words and non-alphanumeric entries
    clean_tokens = [word for word in clean_tokens if word not in stopwords.words("english")]
    clean_tokens = [word for word in clean_tokens if word.isalnum()]

    return clean_tokens


def build_model(X_train, y_train):
    """Building machine learning pipeline with tuned parameters using GridSearchCV:
    - defining pipeline
    - defining parameters for GridSearchCV
    - identifying best parameters
    - fitting pipeline with best paramters
    
    Args:
        X_train (array): input data for training
        y_train (array): output data for training
    Returns:
        pipeline (pipeline): Trained and fitted machine learning pipeline
    """
    # Defining pipeline
    pipeline = Pipeline([('vect', CountVectorizer(tokenizer=tokenize)),
                     ('tfidf', TfidfTransformer()),
                     ('clf', MultiOutputClassifier(KNeighborsClassifier()))
                    ])
    # Defining paramters for GridSearchCV
    parameters = {
    'tfidf__use_idf': (True, False),
    'clf__estimator__leaf_size': [5,10],
    'clf__estimator__n_neighbors': [3],
    'clf__estimator__p': [2]
    }
    
    # Initiating GridSearchCV
    cv = GridSearchCV(pipeline, param_grid=parameters, n_jobs=-1, cv=2, verbose=3)
    
    # Fitting GridSearchCV
    with active_session():
        cv.fit(X_train, y_train)
    
    # Getting and printing best parameters
    best_params = cv.best_params_   
    print(f"Best parameters: {best_params}")
    
    # Setting best parameters
    pipeline.set_params(**best_params)
    
    # Training pipeline with tuned model
    pipeline.fit(X_train, y_train)
    
    return pipeline
    
    


def evaluate_model(model, X_test, Y_test, category_names):
    """ Evaluting model and printing classification report.
    
    Args:
        model (array): trained and fitted model/pipeline
        X_test (array): input data for testing
        Y_test (array): output data for testing
        category_names (list): label names
        
    Returns: None
    """
    
    # Predicting on test data with tuned model
    Y_pred = model.predict(X_test)
    
    # Printing evalution results
    print(classification_report(Y_test, Y_pred, target_names = category_names))


def save_model(model, model_filepath):
    """ Saving model into pickle file """
    # Saving to file in the current working directory
    pickle_filename = model_filepath
    with open(pickle_filename, 'wb') as file:
        pickle.dump(model, file)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model(X_train, Y_train)
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()