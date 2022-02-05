import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from collections import Counter

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
        
    clean_tokens = [word for word in clean_tokens if word not in stopwords.words("english")]
    clean_tokens = [word for word in clean_tokens if word.isalnum()]
    
    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('Disasters', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    clf_counts_top = df.iloc[:,4:].sum().sort_values(ascending = False).head(10).values
    clf_names_top = df.iloc[:,4:].sum().sort_values(ascending = False).head(10).index
    clf_counts_bottom = df.iloc[:,4:].sum().sort_values(ascending = False).tail(10).values
    clf_names_bottom = df.iloc[:,4:].sum().sort_values(ascending = False).tail(10).index
    
    token_list = df[df.weather_related==1].message.apply(tokenize)
    word_list = []

    for row_list in token_list:
        for element in row_list:
            word_list.append(element)
            
    weather_top10 = Counter(word_list).most_common(10)
    weather_top10_words = []
    weather_top10_counts = []
    
    for weather_tuple in weather_top10:
        weather_top10_words.append(weather_tuple[0])
        weather_top10_counts.append(weather_tuple[1])
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
       {
            'data': [
                Bar(
                    x=clf_names_top,
                    y=clf_counts_top
                )
            ],

            'layout': {
                'title': 'Top 10 Message Classifications',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Classification"
                }
            }
        },
       {
            'data': [
                Bar(
                    x=clf_names_bottom,
                    y=clf_counts_bottom
                )
            ],

            'layout': {
                'title': 'Bottom 10 Message Classifications',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Classification"
                }
            }
        },
       {
            'data': [
                Bar(
                    x=weather_top10_words,
                    y=weather_top10_counts
                )
            ],

            'layout': {
                'title': 'Top 10 weather related words',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Word"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3000, debug=True)


if __name__ == '__main__':
    main()