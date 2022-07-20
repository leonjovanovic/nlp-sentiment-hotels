import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from torch import rand


def import_annotated_json() -> pd.DataFrame:
    df1 = pd.read_json(f'data/annotated_reviews_lang_1.json', orient='index')
    df2 = pd.read_json(f'data/annotated_reviews_lang_2.json', orient='index')
    df = pd.concat([df1, df2])
    df = df.drop(axis=1, columns=['category', 'link', 'language'])
    df = df[(df.amenities != 'n/a') & (df.amenities != 'skipped')]
    df.drop_duplicates(inplace=True)
    def map_entities(x):
        for column in range(1, len(x)):
            if x[column] == '':
                x[column] = 0
            elif x[column] == 'n':
                x[column] = 1
            else:
                x[column] = 2
        return x
    df = df.apply(lambda x: map_entities(x), axis=1)
    df.reset_index(drop=True, inplace=True)
    return df


def bag_of_words(df: pd.DataFrame, binary=False) -> pd.DataFrame:     
    vectorizer = CountVectorizer(binary=binary)
    return pd.DataFrame(vectorizer.fit_transform(df).toarray(), columns=vectorizer.get_feature_names_out())


def tokenize(df):
    vectorizer = CountVectorizer()
    vectorizer.fit(df)
    return vectorizer.get_feature_names_out()


def shuffle_dataset(df: pd.DataFrame) -> pd.DataFrame:    
    df = shuffle(df, random_state=0)
    df.reset_index(drop=True, inplace=True)
    return df

def split_dataset(df: pd.DataFrame, train_percent = 0.8, validation_split = True) -> pd.DataFrame:
    train_set, val_test = train_test_split(df, test_size=1-train_percent, shuffle=True, random_state=1)
    train_set.reset_index(drop=True, inplace=True)
    validation_set, test_set = train_test_split(val_test, test_size=0.5, shuffle=False) if validation_split else [None, val_test]
    if validation_set is not None:
        validation_set.reset_index(drop=True, inplace=True)
    test_set.reset_index(drop=True, inplace=True)
    return train_set, validation_set, test_set