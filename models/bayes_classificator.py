import pandas as pd
import numpy as np
from math import log
from preprocess import prepare_dataset
from utils import ModelType, tokenize


class NaiveBayes:
    def __init__(self, type: int) -> None:
        self.type = type
        self.num_classes = 2 if type < ModelType.BOTH else 3
        self.word_in_classes_probabilities = []
        self.classes_probabilities = []


    def calc_probabilities_per_word(self, classes: list) -> list:
        for df in classes:
            df = pd.DataFrame(df.apply(lambda x: sum(x)))[:-1].transpose()
            n = len(df.columns)
            n_sum = df.sum(axis=1)
            self.word_in_classes_probabilities.append(pd.DataFrame(df.apply(lambda x: (x + 1) / (n + n_sum))))


    def calc_probabilities_per_class(self, df: pd.DataFrame) -> pd.DataFrame:     
        classes = []     
        for i in range(self.num_classes):   
            tmp_class = df.loc[df[df.columns[-1]] == i]
            tmp_class.reset_index(drop=True, inplace=True)
            classes.append(tmp_class)
            self.classes_probabilities.append(len(tmp_class)/len(df))
        self.calc_probabilities_per_word(classes)


    def reset_model(self):
        self.word_in_classes_probabilities = []
        self.classes_probabilities = []


    def train(self, input_data, output_data):
        self.reset_model()
        df_train = prepare_dataset(input_data, output_data, self.type)
        self.calc_probabilities_per_class(df_train)


    def compute(self, review):
        class_sums = [log(class_prob) for class_prob in self.classes_probabilities]
        all_words = list(self.word_in_classes_probabilities[0].columns)
        for word in review:
            if word in all_words:
                for idx, word_in_class_prob in enumerate(self.word_in_classes_probabilities):
                    class_sums[idx] += log(float(word_in_class_prob[word]))
        return np.argmax(np.array(class_sums))

    
    def test(self, input_data, output_data):
        input_data = input_data.apply(lambda x: tokenize([x.hotel_review]), axis=1)
        df_test = prepare_dataset(input_data, output_data, self.type)
        df_test['prediction'] = pd.DataFrame(df_test.apply(lambda x: self.compute(x[0]), axis=1))
        return len(df_test[df_test['prediction'] != df_test[df_test.columns[-2]]])/len(df_test), \
               len(df_test[df_test['prediction'] == df_test[df_test.columns[-2]]])/len(df_test)*100.0


class NaiveBayesCombined:
    def __init__(self) -> None:
        self.bayes_category = NaiveBayes(ModelType.CATEGORY)
        self.bayes_sentiment = NaiveBayes(ModelType.SENTIMENT)

    def train(self, input_data, output_data):
        self.bayes_category.train(input_data, output_data)
        self.bayes_sentiment.train(input_data, output_data)

    def test(self, input_data, output_data):
        input_data = input_data.apply(lambda x: tokenize([x.hotel_review]), axis=1)
        prediction = pd.DataFrame(input_data).apply(lambda x: self.compute(x[0]), axis=1)
        df_test = pd.concat([input_data.rename('hotel_review'), output_data, prediction.rename('prediction')], axis=1)
        return len(df_test[df_test['prediction'] != df_test[df_test.columns[-2]]])/len(df_test), \
               len(df_test[df_test['prediction'] == df_test[df_test.columns[-2]]])/len(df_test)*100.0

    def compute(self, review):
        if self.bayes_category.compute(review):
            return self.bayes_sentiment.compute(review) + 1
        else:
            return 0