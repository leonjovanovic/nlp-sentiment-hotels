import string
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


class ModelType:
    CATEGORY = 0
    SENTIMENT = 1
    BOTH = 2

    @staticmethod
    def map_value_to_string(value: int) -> str:
        if value == 0:
            return 'category'
        elif value == 1:
            return 'sentiment'
        elif value == 2:
            return 'both'
        else:
            raise Exception('Bad value for ModelType enum')


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


def bag_of_words(df: pd.Series, binary=False) -> pd.DataFrame:
    vectorizer = CountVectorizer(binary=binary)
    return pd.DataFrame(vectorizer.fit_transform(df).toarray(), columns=vectorizer.get_feature_names_out())


def tokenize(df):
    vectorizer = CountVectorizer()
    vectorizer.fit(df)
    return vectorizer.get_feature_names_out()


def shuffle_dataset(df: pd.DataFrame, random_state=0) -> pd.DataFrame:
    df = shuffle(df, random_state=random_state)
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


def split_dataset_into_two(df: pd.DataFrame, test_val_index, k_fold=10):
    df = shuffle_dataset(df)

    slice_len = int(len(df)/k_fold)
    test_set = df.iloc[test_val_index*slice_len:((test_val_index+1)*slice_len if test_val_index != k_fold-1 else len(df)), :]
    test_set.reset_index(drop=True, inplace=True)

    train_set_1st = df.iloc[0:test_val_index*slice_len, :]
    train_set_2nd = df.iloc[((test_val_index+1)*slice_len if test_val_index != k_fold-1 else len(df)): len(df), :]
    train_set = pd.concat([train_set_1st, train_set_2nd])
    train_set.reset_index(drop=True, inplace=True)

    return train_set, test_set
