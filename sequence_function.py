import numpy as np
from data_extracting import data_18_19
import json


all_data = data_18_19
all_data.fillna(0, inplace=True)
all_data['const'] = 1


def get_all_points(all_players_data):
    predicted = {}
    for i in all_players_data.keys():
        predicted[i] = predict_player_points(all_players_data[i])
    return predicted


def generate_sequnces(all_data):
    season1 = all_data['total_points_s1']
    season2 = all_data['total_points_s2']
    season3 = all_data['total_points_s3']
    name = all_data['name']

    seq_dict = {}
    for i in range(len(name)):
        seq_dict[name[i]] = [int(season1[i]), int(season2[i]), int(season3[i])]

    return seq_dict


def predict_player_points(data):
    for i in range(len(data)):
        data[i] = [i, data[i]]
    X = np.asmatrix(data)[:, 0]
    y = np.asmatrix(data)[:, 1]

    def J(X, y, theta):
        theta = np.asmatrix(theta).T
        m = len(y)
        predictions = X * theta
        sqError = np.power((predictions - y), [2])
        return 1 / (2 * m) * sum(sqError)

    dataX = np.asmatrix(data)[:, 0:1]
    X = np.ones((len(dataX), 2))
    X[:, 1:] = dataX

    def gradient(X, y, alpha, theta, iters):
        J_history = np.zeros(iters)
        m = len(y)
        theta = np.asmatrix(theta).T
        for i in range(iters):
            h0 = X * theta
            delta = (1 / m) * (X.T * h0 - X.T * y)
            theta = theta - alpha * delta
            J_history[i] = J(X, y, theta.T)
        return J_history, theta

    theta = np.asmatrix([np.random.random(), np.random.random()])
    alpha = 0.01
    iters = 10000

    print(
        '\n== Model summary ==\nLearning rate: {}\nIterations: {}\nInitial theta: {}\nInitial J: {:.2f}\n'.format(alpha,
                                                                                                                  iters,
                                                                                                                  theta,
                                                                                                                  J(X,
                                                                                                                    y,
                                                                                                                    theta).item()))

    # print('Training the model... ')
    # this actually trains our model and finds the optimal theta value
    J_history, theta_min = gradient(X, y, alpha, theta, iters)
    # print('Done.')
    print('\nThe modelled prediction function is:\ny = {:.2f} * x + {:.2f}'.format(theta_min[1].item(),
                                                                                  theta_min[0].item()))

    def predict(pop):
        return [1, pop] * theta_min

    p = len(data)
    final_prediction = predict(p).item()
    return final_prediction



points = generate_sequnces(all_data)

print(points)


predictions = get_all_points(points)

with open("predictions.txt", 'w') as f:
    data_to_write = json.dumps(predictions)
    f.write(data_to_write)


print(predictions)