import difflib
import math
import numpy
import validation
import operator
import process_data


def euclidean_distance(row1, row2):
    distance = 0.0
    for i in range(len(row1) - 1):
        distance += (row1[i] - row2[i]) ** 2
    return math.sqrt(distance)


def get_neighbors(train, test_row, num_neighbors):
    distances = list()
    for train_row in train:
        dist = euclidean_distance(test_row, train_row[0])
        distances.append((train_row, dist))
    distances.sort(key=lambda tup: tup[1])
    neighbors = list()
    for i in range(num_neighbors):
        neighbors.append(distances[i][0])
    return neighbors


def get_most_frequent(labels):
    labels_dict = dict()
    for label in labels:
        if label not in labels_dict.keys():
            labels_dict[label] = 0
        else:
            labels_dict[label] = labels_dict[label]+1
    return max(labels_dict, key=labels_dict.get)


def predict_classification(train, test_row, num_neighbors):
    neighbors = get_neighbors(train, test_row, num_neighbors)
    labels = [row[1] for row in neighbors]
    # prediction = max(set(labels), key=labels.count)
    prediction = get_most_frequent(labels)
    return prediction


def k_nearest_neighbors(train, test, num_neighbors):
    predictions = list()
    for row in test:
        output = predict_classification(train, row, num_neighbors)
        predictions.append(output)
    return numpy.array(predictions)


if __name__ == '__main__':
    X_train, y_train, X_test, y_test, length = process_data.get_processed_data()
    # knn
    train_data = []
    for i in range(len(X_train)):
        train_data.append((X_train[i], y_train[i]))

    k_range = 25
    print(f"Search for optimal k value under range 3 to {k_range}")
    k_accuracy = dict()

    for i in range(3, k_range):
        print(i)
        temp = k_nearest_neighbors(train_data, X_test, i)
        accuracy = numpy.mean(temp == y_test)
        k_accuracy[i] = accuracy

    K_val = max(k_accuracy, key=k_accuracy.get)
    print(f"Best k value is: {K_val}")
    a = k_nearest_neighbors(train_data, X_test, 50)
    test_accuracy = numpy.mean(a == y_test)

    print("-----------Accuracy Result-----------")
    print("The test accuracy is: " + str(test_accuracy))
    w, e = validation.get_f1_score(y_test, a)
    print(w)
    print(e)
