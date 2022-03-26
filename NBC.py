import math
import numpy
from matplotlib import pyplot as plt

import validation
import process_data


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
        self.num_of_features = feature_types
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


def run():
    X_train, y_train, X_test, y_test, length, x_fill = process_data.get_processed_data()
    nbc = NBC(feature_types=length, num_classes=5)
    nbc.fit(X_train, y_train)
    a = nbc.predict(X_test)
    test_accuracy = numpy.mean(a == y_test)
    print("-----------Accuracy Result-----------")
    print("The test accuracy is: " + str(test_accuracy))
    validation.show_f1_score(y_test, a)

    return nbc.predict(x_fill)
