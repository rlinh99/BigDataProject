import math
import numpy
import validation
import operator
import process_data


class KNN:
    def get_distance(datapoint1, datapoint2):
        distance = 0
        for x in range(len(X_train)):
            distance += (datapoint1[x] - datapoint2[x]) ** 2

        final_distance = math.sqrt(distance)
        return final_distance

    def get_K_neighbors(X_train, y_test, K):
        ## 1. get all distances
        all_distances = []
        for i in range(len(X_train)):
            distance = KNN.get_distance(X_train[i], y_test)
            all_distances.append((X_train[i], distance))

        ### 2. sort all distances
        all_distances.sort(key=operator.itemgetter(1))

        ### 3. pickup the K nearest neighbours
        K_neighbors = []
        for k in range(K):
            K_neighbors.append(all_distances[k][0])
        return K_neighbors


    def get_prediction(K_neighbors):
        category_votes = {}
        for i in range(len(K_neighbors)):
            category = K_neighbors[i][-1]
            if category in category_votes:
                category_votes[category] += 1
            else:
                category_votes[category] = 1
        sort_category_votes = sorted(category_votes.items(), key=operator.itemgetter(1), reverse=True)

        most_vote = sort_category_votes[0][0]
        return numpy.array(most_vote)


if __name__ == '__main__':
    X_train, y_train, X_test, y_test, length = process_data.get_processed_data()
    # knn
    K = 3
    a = KNN.get_prediction(X_test)
    test_accuracy = numpy.mean(a == y_test)

    print("-----------Accuracy Result-----------")
    print("The test accuracy is: " + str(test_accuracy))
    w, e = validation.get_f1_score(y_test, a)
    print(w)
    print(e)
