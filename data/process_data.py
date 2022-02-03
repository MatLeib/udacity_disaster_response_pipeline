# Imports
import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """Loading two datasets, merging them and returning merged DataFrame."""
    # Loading datasets
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    # Merging datasets
    df = pd.merge(messages, categories, on="id", how='outer')
    
    return df 


def clean_data(df):
    """Cleaning data contains following steps:
    - Creating 36 new columns of the categories column of original dataframe
    - Filling new columns with numeric values 0 or 1
    - Dropping original categories column
    - Dropping duplicate rows
    - Returning cleaned dataframe
    """
    # Creating a dataframe of the 36 individual category columns
    categories = df.categories.str.split(pat=';', expand=True)
    
    # Extracting a list of new column names for categories
    category_colnames = categories.iloc[0,:].str.slice(start=0, stop=-2).tolist()
    
    # Renaming the columns of `categories`
    categories.columns = category_colnames
    
    # Converting category values to just numbers 0 or 1
    for column in categories:
        # Setting each value to be the last character of the string
        categories[column] = categories[column].str.slice(start=-1)

        # Converting column from string to numeric
        categories[column] = pd.to_numeric(categories[column], downcast='integer')

    # Dropping the original categories column from `df`
    df.drop(columns=['categories'], inplace=True)
    
    # Concatenating the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1, join="inner")
    
    # Dropping duplicates
    df.drop_duplicates(inplace=True)
    
    # Replacing values "2" with "1"
    df.replace(2,1,inplace=True)
    
    # Dropping columns with only one distinct value
    for col in df.columns:
        if len(df[col].unique()) == 1:
            print(f"    Column '{col}' has only one distinct value and therefore will be dropped.")
            df.drop(col,inplace=True,axis=1)
    
    return df


def save_data(df, database_filename):
    """Saving DataFrame as SQL database"""
    # Creating SQL engine
    engine = create_engine(f"sqlite:///{database_filename}")
    
    # Saving DataFrame to SQL database, overwriting table in case already existing
    df.to_sql('Disasters', engine, index=False, if_exists='replace')  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()