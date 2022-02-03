# Importing libraries
import sys

import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])

import re
import pandas as pd
from sqlalchemy import create_engine
import pickle

from nltk.tokenize import word_tokenize
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
    """ DOCSTRING """
    # load data from database
    engine = create_engine('sqlite:///'+str(database_filepath))
    df = pd.read_sql_table('Disasters', engine)
    X = df.message.values
    y = df.iloc[:,4:].values
    category_names = df.iloc[:,4:].columns.tolist()
    
    return  X, y, category_names


def tokenize(text):
    """ DOCSTRING """
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(lemmatizer.lemmatize(tok), pos='v').lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model(X_train, y_train):
    """ DOCSTRING """
    pipeline = Pipeline([('vect', CountVectorizer(tokenizer=tokenize)),
                     ('tfidf', TfidfTransformer()),
                     ('clf', MultiOutputClassifier(KNeighborsClassifier()))
                    ])
    
    parameters = {
    'tfidf__use_idf': (True, False),
#     'clf__estimator__leaf_size': [9,10],
    'clf__estimator__n_neighbors': [4,5],
    'clf__estimator__p': [1,2]
    }

    cv = GridSearchCV(pipeline, param_grid=parameters, n_jobs=-1, cv=2, verbose=3)

    cv.fit(X_train, y_train)
 
    best_params = cv.best_params_
    
    file = open("models/best_params.txt", "w") 
    file.write(str(best_params)) 
    file.close()
    
    # Setting best parameters
    pipeline.set_params(**best_params)
    
    # Training pipeline with tuned model
    pipeline.fit(X_train, y_train)
    
    return pipeline
    
    


def evaluate_model(model, X_test, Y_test, category_names):
    """ DOCSTRING """
    
    # Predicting on test data with tuned model
    Y_pred = model.predict(X_test)
    
    print(classification_report(Y_test, Y_pred, target_names= category_names))


def save_model(model, model_filepath):
    """ DOCSTRING """
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