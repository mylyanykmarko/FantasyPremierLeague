import pandas as pd
import json
import os
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

TARGET = 142.07


def get_json_data(filename):
    with open(os.path.join("raw_players", filename), 'r') as f:
        data = json.load(f)
    return data['elements']


def get_cvs_data(filename):
    data = pd.read_csv(os.path.join("raw_players", filename))
    return data


def get_useful_data(all_data):
    """
    Dropping useless columns and prepare data
    :param all_data: dict()
    :return: pandas.DataFrame()
    """
    if isinstance(all_data, list):
        data = pd.DataFrame.from_dict(all_data)
    else:
        data = all_data
    useful_columns = ["second_name", "first_name", "element_type", "total_points", "id", "in_dreamteam", "minutes",
                      "now_cost", "team", "creativity", "goals_scored", "influence"]
    data = data.filter(useful_columns)
    data['first_name'] = data['first_name'] + ' ' + data['second_name']
    data = data.drop('second_name', 1)
    data.rename(columns={'first_name': 'name'}, inplace=True)

    return data


def split_players_by_position(data):
    """
    Split players by positions
    :param data: pandas.DataFrame()
    :return:  4 x pandas.DataFrame()
    """
    goalkeepers = data[data['element_type'] == 1]
    defenders = data[data['element_type'] == 2]
    midfielders = data[data['element_type'] == 3]
    forwards = data[data['element_type'] == 4]
    return goalkeepers, defenders, midfielders, forwards


data_16_17 = get_cvs_data('players_raw_20162017.csv')
data_16_17 = get_useful_data(data_16_17)
data_16_17['ppp_s1'] = data_16_17['total_points']/(data_16_17['now_cost']/10)
data_16_17['ppm_s1'] = data_16_17['total_points']/data_16_17['minutes']
data_16_17.rename(columns={'total_points': 'total_points_s1'}, inplace=True)
data_16_17.rename(columns={'now_cost': 'price_s1'}, inplace=True)
data_16_17.rename(columns={'minutes': 'minutes_s1'}, inplace=True)


data_17_18 = get_cvs_data('players_raw_20172018.csv')
data_17_18 = get_useful_data(data_17_18)
data_17_18['ppp_s2'] = data_17_18['total_points']/(data_17_18['now_cost']/10)
data_17_18['ppm_s2'] = data_17_18['total_points']/data_17_18['minutes']
data_17_18.rename(columns={'total_points': 'total_points_s2'}, inplace=True)
data_17_18.rename(columns={'now_cost': 'price_s2'}, inplace=True)
data_17_18.rename(columns={'minutes': 'minutes_s2'}, inplace=True)


data_18_19 = get_json_data('players_raw_20182019.json')
data_18_19 = get_useful_data(data_18_19)
data_18_19['ppp_s3'] = data_18_19['total_points']/(data_18_19['now_cost']/10)
data_18_19['ppm_s3'] = data_18_19['total_points']/data_18_19['minutes']
data_18_19.rename(columns={'total_points': 'total_points_s3'}, inplace=True)
data_18_19.rename(columns={'now_cost': 'price_s3'}, inplace=True)
data_18_19.rename(columns={'minutes': 'minutes_s3'}, inplace=True)

# print(data_16_17)


def join_all_data(data1, data2):
    names_base = data_18_19['name']
    new_names = data2['name']

    total = pd.merge(data1, data2, on='name', how='left')
    return total


def get_correlation(data):
    data.insert(0, "target", data.pop("target"))
    data = data.drop("name", 1)
    data = data.drop("id", 1)
    data.fillna(0, inplace=True)
    X = data.iloc[:, 1:]
    y = data.iloc[:, 0]
    bestfeatures = SelectKBest(score_func=chi2, k=5)
    fit = bestfeatures.fit(X, y)
    dfscores = pd.DataFrame(fit.scores_)
    dfcolumns = pd.DataFrame(X.columns)
    # concat two dataframes for better visualization
    featureScores = pd.concat([dfcolumns, dfscores], axis=1)
    featureScores.columns = ['Specs', 'Score']  # naming the dataframe columns
    print(featureScores.nlargest(10, 'Score'))

test_column = data_18_19['total_points_s3']
target_column = []

for i in test_column:
    if i > TARGET:
        target_column.append(1)
    else:
        target_column.append(0)

new_data = data_18_19
new_data['target'] = target_column
get_correlation(new_data)

data_17_18 = join_all_data(data_17_18, data_16_17)
data_18_19 = join_all_data(data_18_19, data_17_18)
data_18_19 = data_18_19.drop('team_y', 1)
data_18_19 = data_18_19.drop('team_x', 1)
data_18_19 = data_18_19.drop('element_type_x', 1)
data_18_19 = data_18_19.drop('element_type_y', 1)
data_18_19 = data_18_19.drop('in_dreamteam_x', 1)
data_18_19 = data_18_19.drop('in_dreamteam_y', 1)
data_18_19 = data_18_19.drop('id_x', 1)
data_18_19 = data_18_19.drop('id_y', 1)



# data_18_19 = data_18_19.drop('in_dreamteam', 1)
points = data_18_19.sort_values('total_points_s3')
# print(points['Gylfi Sigurdsson'])
# print(points[['name', 'total_points_s3']])
# print(points.loc[points['name'] == 'Sergio Agüero'])
# print(points['total_points_s3'].tail(100))
# print(points['total_points_s3'].tail(100).mean())

# data_18_19[''] = data_18_19
# print(data_18_19['team'])

# print(data_18_19['element_type'])
# print(list(data))
