import pandas as pd
import numpy as np
from math import log
from preprocess import prepare_dataset
from utils import ModelType
from sklearn.metrics import f1_score


class NaiveBayes:
    def __init__(self, type: int) -> None:
        if type > ModelType.BOTH:
            raise Exception('Bad value for ModelType enum')
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


    def train(self, train_input_data, train_output_data, test_input_data=None, test_output_data=None):
        self.reset_model()
        df_train = prepare_dataset(train_input_data, train_output_data, self.type)
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
        df_test = prepare_dataset(input_data, output_data, self.type)
        df_test['prediction'] = pd.DataFrame(df_test.apply(lambda x: self.compute(x[0]), axis=1))
        return len(df_test[df_test['prediction'] != df_test[df_test.columns[-2]]])/len(df_test), \
            f1_score(df_test[df_test.columns[-2]], df_test['prediction'], average='macro')


class NaiveBayesCombined:
    def __init__(self) -> None:
        self.bayes_category = NaiveBayes(ModelType.CATEGORY)
        self.bayes_sentiment = NaiveBayes(ModelType.SENTIMENT)

    def train(self, train_input_data, train_output_data, test_input_data=None, test_output_data=None):
        self.bayes_category.train(train_input_data, train_output_data)
        self.bayes_sentiment.train(train_input_data, train_output_data)

    def test(self, input_data, output_data):
        prediction = input_data.apply(lambda x: self.compute(x[0]), axis=1)
        df_test = pd.concat([output_data, prediction], axis=1)
        return len(df_test[df_test[df_test.columns[-1]] != df_test[df_test.columns[-2]]])/len(df_test), \
            f1_score(df_test[df_test.columns[-2]], df_test[df_test.columns[-1]], average='macro')

    def compute(self, review):
        if self.bayes_category.compute(review):
            return self.bayes_sentiment.compute(review) + 1
        else:
            return 0