import pandas as pd
import numpy as np
import os

from sequence_function import predictions, generate_sequnces, get_all_points, predict_player_points

from data_extracting import data_18_19, split_players_by_position

BUDGET = 100
NUMBER_OF_PLAYERS = 15
POSITIONS = [2, 5, 5, 3]

# all_data['']
all_data = data_18_19
all_data.fillna(0, inplace=True)
all_data['const'] = 1


all_data['ppp_s4'] = (all_data['ppp_s1'] + all_data['ppp_s2'] + all_data['ppp_s3'])/3

goalkeepers, defenders, midfielders, forwards = split_players_by_position(all_data)

goalkeepers.sort_values('ppp_s4')
defenders.sort_values('ppp_s4')
midfielders.sort_values('ppp_s4')
forwards = forwards.sort_values(by='ppp_s4')

print(forwards[0][1])


# goalkeepers, defenders, midfielders, forwards = split_players_by_position(all_data)


def predict_future_points(all_data):
    pass


def predict_future_minutes(all_data):
    pass


# predicted = predict_future_minutes(all_data)
# predicted = predict_future_points(all_data)

# goalkeepers, defenders, midfielders, forwards = split_players_by_position(predicted)

def pick_optimal_team(data):
    pass

