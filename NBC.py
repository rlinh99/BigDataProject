import nltk
from bs4 import BeautifulSoup
from nltk import SnowballStemmer
from nltk.corpus import stopwords
import textblob as TextBlob
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
import math
import numpy
import data_loader
import validation
import re

nltk.download('stopwords')
stopwords = set(stopwords.words('english'))


def get_mean(lst):
    return sum(lst) / len(lst)


def get_variance(lst):
    diff = list()
    mean = get_mean(lst)

    for item in lst:
        diff.append(pow((item - mean), 2))

    variance = get_mean(diff)
    if variance == 0:
        return 1e-6
    return variance


def get_sd(lst):
    return math.sqrt(get_variance(lst))


# gaussian distribution function
def get_gaussian(data, mean, sd):
    return (1 / (math.sqrt(2 * math.pi) * sd)) * \
           math.exp(-((data - mean) ** 2 / (2 * sd ** 2)))


class NBC:
    def __init__(self, feature_types=None, num_classes=0):
        self.classes = dict()
        self.trained = dict()
        self.XTrain = None
        self.yTrain = None
        self.feature_types = feature_types
        self.num_of_features = len(feature_types)
        self.num_of_classes = num_classes

    def classify_train_data(self):
        for i, target in enumerate(self.XTrain):
            if self.yTrain[i] not in self.classes:
                self.classes[self.yTrain[i]] = list()
            self.classes[self.yTrain[i]].append(target)

    # calculate mean and sd for each feature
    def handle_features(self, class_data):
        temp = list()

        for i in range(self.num_of_features):
            features = []
            for item in class_data:
                features.append(item[i])
            temp.append((len(features), get_mean(features), get_variance(features)))
        return temp

    def get_class_probabilities(self):
        probabilities = dict()
        for key, value in self.trained.items():
            probabilities[key] = value[0][0] / len(self.XTrain)
        return probabilities

    def get_probability(self, source):
        probabilities = dict()
        for key, value in self.trained.items():
            probabilities[key] = value[0][0] / len(self.XTrain)
            for index in range(len(value)):
                _, mean, sd = value[index]
                probabilities[key] = probabilities[key] \
                                     * get_gaussian(source[index], mean, sd)
        return probabilities

    def get_prediction(self, source):
        actual_prob = None
        predicted_class = None
        prob_results = self.get_probability(source)

        for key, value in prob_results.items():
            if actual_prob is None:
                actual_prob = value
                predicted_class = key
            if value > actual_prob:
                actual_prob = value
                predicted_class = key
        return predicted_class

    # fit method
    def fit(self, XTrain=None, yTrain=None):
        self.XTrain = XTrain
        self.yTrain = yTrain
        self.trained = dict()
        self.classes = dict()
        self.classify_train_data()

        for key, value in self.classes.items():
            self.trained[key] = self.handle_features(value)

        print("Naive Bayes Classifier Trained.")
        # Print class probabilities
        print("------------Class Probability Results-------------")
        print("The class probability is: ")
        print(self.get_class_probabilities())
        print("The conditional distribution of features for each class is:"
              " (Count, Mean, Standard Deviation)")
        for i, _ in enumerate(self.trained):

            print("Class " + str(i))
            # for item in self.trained[i]:
            #     print(item)

    # method to predict
    def predict(self, XTest):
        if len(self.trained) == 0:
            print("Please fit first")
            return False

        predictions = list()
        for item in XTest:
            predicted_class = self.get_prediction(item)
            predictions.append(predicted_class)
        return numpy.array(predictions)

def convert_comment(entries):
    converted = []
    for entry in entries:
        result = TextBlob.TextBlob(entry)
        converted.append(result.sentiment.polarity)
    return converted


def review_to_words(raw_review):
    # 1. Delete HTML
    review_text = BeautifulSoup(raw_review, 'html.parser').get_text()
    # 2. Make a space
    letters_only = re.sub('[^a-zA-Z]', ' ', review_text)
    # 3. lower letters
    words = review_text.lower().split()
    # 5. Stopwords
    meaningful_words = [w for w in words if not w in stopwords]
    # 6. Stemming
    stemming_words = [stemmer.stem(w) for w in meaningful_words]
    # 7. space join words

    return( ' '.join(stemming_words))

if __name__ == '__main__':
    train_features = ['drugName', 'condition', 'reviewComment', 'usefulCount', 'sideEffects']
    train = data_loader.train_data
    train = train.dropna(how = 'any', axis = 0)
    test = data_loader.test_data
    X_train_raw, y_train_raw = train[train_features], train['rating']
    X_test_raw, y_test_raw = test[train_features], test['rating']
    print("Data Loaded.")

    # clean review
    # remove stemmer
    stemmer = SnowballStemmer('english')
    reviews = X_train_raw['reviewComment']
    reviews_t = X_test_raw['reviewComment']

    X_train_raw['clean_comment'] = reviews.apply(review_to_words)
    X_test_raw['clean_comment'] = reviews_t.apply(review_to_words)

    clean_reviews = X_train_raw['clean_comment']
    clean_reviews_t = X_test_raw['clean_comment']

    # clean word count
    X_train_raw['word_count'] = clean_reviews.apply(lambda x: len(str(x).split()))
    X_test_raw['word_count'] = clean_reviews_t.apply(lambda x: len(str(x).split()))

    # handle comment value
    X_train_raw['comment_val'] = convert_comment(reviews)
    X_test_raw['comment_val'] = convert_comment(reviews_t)

    X_train_raw['clean_comment_val'] = convert_comment(clean_reviews)
    X_test_raw['clean_comment_val'] = convert_comment(clean_reviews_t)

    # categorical data
    le = LabelEncoder()
    le.fit(X_train_raw['sideEffects'])

    X_train_raw['sideEffects_val'] = le.transform(X_train_raw['sideEffects'])
    X_test_raw['sideEffects_val'] = le.transform(X_test_raw['sideEffects'])

    # le2.fit(X_train_raw['drugName'])
    # X_train_raw['drugName_val'] = le2.transform(X_train_raw['drugName'])
    # X_test_raw['drugName_val'] = le2.transform(X_test_raw['drugName'])
    eva_labels = ['sideEffects_val', 'comment_val', 'clean_comment_val']
    X_train = X_train_raw[eva_labels].to_numpy()
    X_train = preprocessing.scale(X_train)

    X_test = X_test_raw[eva_labels].to_numpy()
    X_test = preprocessing.scale(X_test)

    y_train = y_train_raw.to_numpy()
    y_test = y_test_raw.to_numpy()

    print("Data Handling Done.")
    nbc = NBC(feature_types=['r', 'r', 'r'], num_classes=5)

    nbc.fit(X_train, y_train)
    a = nbc.predict(X_test)
    test_accuracy = numpy.mean(a == y_test)
    print("-----------Accuracy Result-----------")
    print("The test accuracy is: " + str(test_accuracy))
    w, e = validation.get_f1_score(y_test, a)
    print(w)
    print(e)
