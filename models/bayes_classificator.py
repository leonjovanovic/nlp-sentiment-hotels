import pandas as pd
import numpy as np
from math import log
import utils


class ModelType:
    CATEGORY = 0
    SENTIMENT = 1
    BOTH = 2


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


    def train(self, input_data, output_data):
        df_train = self.prepare_dataset(input_data, output_data)
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
        df_test = self.prepare_dataset(input_data, output_data)
        df_test['prediction'] = pd.DataFrame(df_test.apply(lambda x: self.compute(x[0]), axis=1))
        return df_test


    def prepare_dataset(self, input_data, output_data):
        input_data = pd.DataFrame(input_data)
        if self.type == ModelType.CATEGORY:
            return pd.concat([input_data, output_data.apply(lambda x: 1 if x == 2 else x)], axis=1)
        elif self.type == ModelType.SENTIMENT:
            data = pd.concat([input_data, output_data], axis=1)
            data = data[data[data.columns[-1]] != 0]
            data[data.columns[-1]] = data[data.columns[-1]].apply(lambda x: x - 1)
            data.reset_index(drop=True, inplace=True)
            return data
        return pd.concat([input_data, output_data], axis=1)


class NaiveBayesCombined:
    def __init__(self) -> None:
        self.bayes_category = NaiveBayes(ModelType.CATEGORY)
        self.bayes_sentiment = NaiveBayes(ModelType.SENTIMENT)

    def train(self, input_data, output_data):
        self.bayes_category.train(input_data, output_data)
        self.bayes_sentiment.train(input_data, output_data)

    def test(self, input_data, output_data):
        prediction = pd.DataFrame(input_data).apply(lambda x: self.compute(x[0]), axis=1)
        return pd.concat([input_data.rename('hotel_review'), output_data, prediction.rename('prediction')], axis=1)

    def compute(self, review):
        if self.bayes_category.compute(review):
            return self.bayes_sentiment.compute(review) + 1
        else:
            return 0