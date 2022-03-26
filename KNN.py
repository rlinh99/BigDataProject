import math
import numpy
import validation
import process_data

# switch between distance type
dist_type = 'eucd'


# dist_type = 'manhattan'
def man_distance(row1, row2):
    return sum(abs(val1 - val2) for val1, val2 in zip(row1, row2))


def euc_distance(row1, row2):
    distance = 0.0
    for i in range(len(row1) - 1):
        distance += (row1[i] - row2[i]) ** 2
    return math.sqrt(distance)


def calculate_dist(row1, row2):
    if dist_type == 'manhattan':
        return man_distance(row1, row2)
    return euc_distance(row1, row2)


def get_neighbors(train_data, test_row, k):
    all_distances = list()
    for row in train_data:
        dist = calculate_dist(test_row, row[0])
        all_distances.append((row, dist))
    all_distances.sort(key=lambda tup: tup[1])
    neighbors = list()
    for i in range(k):
        neighbors.append(all_distances[i][0])
    return neighbors


def get_most_frequent(labels):
    labels_dict = dict()
    for label in labels:
        if label not in labels_dict.keys():
            labels_dict[label] = 0
        else:
            labels_dict[label] = labels_dict[label] + 1
    return max(labels_dict, key=labels_dict.get)


def get_prediction(train, test_row, num_neighbors):
    neighbors = get_neighbors(train, test_row, num_neighbors)
    labels = [row[1] for row in neighbors]
    prediction = get_most_frequent(labels)
    return prediction


def k_Nearest_Neighbors(train, test, k):
    predictions = list()
    for row in test:
        output = get_prediction(train, row, k)
        predictions.append(output)
    return numpy.array(predictions)


def run():
    X_train, y_train, X_test, y_test, length , x_fill= process_data.get_processed_data()
    # knn
    train_data = []
    for i in range(len(X_train)):
        train_data.append((X_train[i], y_train[i]))

    k_range = 50
    print(f"Search for optimal k value under range 30 to {k_range}")
    k_accuracy = dict()

    for i in range(30, k_range):
        print(i)
        temp = k_Nearest_Neighbors(train_data, X_test, i)
        accuracy = numpy.mean(temp == y_test)
        k_accuracy[i] = accuracy

    K_val = max(k_accuracy, key=k_accuracy.get)
    print(f"Best k value is: {K_val}")

    a = k_Nearest_Neighbors(train_data, X_test, K_val)
    test_accuracy = numpy.mean(a == y_test)
    print("-----------Accuracy Result-----------")
    print("The test accuracy is: " + str(test_accuracy))
    validation.show_f1_score(y_test, a)
    result = k_Nearest_Neighbors(train_data, x_fill, K_val)
    return result
