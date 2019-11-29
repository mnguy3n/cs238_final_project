from connect4 import Connect4Board
from connect4 import GameState
from connect4 import Player
from basis_function import basis_vector
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

import numpy as np
import pandas as pd
import random
import sys
import time

def ingest_data():
  data_20 = pd.read_csv("game_data/episode_data_20.csv")
  data_100 = pd.read_csv("game_data/episode_data_100.csv")
  data_500 = pd.read_csv("game_data/episode_data_500.csv")
  print("100 episode shape:", data_100.shape)
  print("500 episode shape:", data_500.shape)
  data = pd.concat([data_100, data_500], ignore_index=True)
  return data, data_20

if __name__ == "__main__":
  start_time = time.time()

  # Prepare data
  train_data, validation_data = ingest_data()
  print("Training data shape:", train_data.shape)
  print("Validation data shape:", train_data.shape)
  feature_columns = [
    "opponent_2_out_of_4",
    "opponent_3_out_of_4",
    "opponent_3_out_of_5",
    "opponent_num_consecutive_possible_wins_in_col_0",
    "opponent_num_consecutive_possible_wins_in_col_1",
    "opponent_num_consecutive_possible_wins_in_col_2",
    "opponent_num_consecutive_possible_wins_in_col_3",
    "opponent_num_consecutive_possible_wins_in_col_4",
    "opponent_num_consecutive_possible_wins_in_col_5",
    "opponent_num_consecutive_possible_wins_in_col_6",
    "opponent_num_possible_wins_in_col_0",
    "opponent_num_possible_wins_in_col_1",
    "opponent_num_possible_wins_in_col_2",
    "opponent_num_possible_wins_in_col_3",
    "opponent_num_possible_wins_in_col_4",
    "opponent_num_possible_wins_in_col_5",
    "opponent_num_possible_wins_in_col_6",
    "opponent_win",
    "player_2_out_of_4",
    "player_3_out_of_4",
    "player_3_out_of_5",
    "player_num_consecutive_possible_wins_in_col_0",
    "player_num_consecutive_possible_wins_in_col_1",
    "player_num_consecutive_possible_wins_in_col_2",
    "player_num_consecutive_possible_wins_in_col_3",
    "player_num_consecutive_possible_wins_in_col_4",
    "player_num_consecutive_possible_wins_in_col_5",
    "player_num_consecutive_possible_wins_in_col_6",
    "player_num_possible_wins_in_col_0",
    "player_num_possible_wins_in_col_1",
    "player_num_possible_wins_in_col_2",
    "player_num_possible_wins_in_col_3",
    "player_num_possible_wins_in_col_4",
    "player_num_possible_wins_in_col_5",
    "player_num_possible_wins_in_col_6",
    "player_win"
  ]
  print("Number of features:", len(feature_columns))
  train_x = train_data[feature_columns]
  train_y = train_data["reward"]
  validation_x = validation_data[feature_columns]
  validation_y = validation_data["reward"]

  # Train model
  model = LinearRegression()
  model.fit(train_x, train_y)

  # Score model
  train_pred = model.predict(train_x)
  validation_pred = model.predict(validation_x)
  print("MSE on training data: %.4f" %(mean_squared_error(train_y, train_pred)))
  print("R^2 on training data: %.4f" %(r2_score(train_y, train_pred))) #1.0 is perfect prediction
  print("MSE on validationing data: %.4f" %(mean_squared_error(validation_y, validation_pred)))
  print("R^2 on validationing data: %.4f" %(r2_score(validation_y, validation_pred))) #1.0 is perfect prediction

  # Output params
  print("Number of coefficients:", len(model.coef_))
  print("Coefficients:", model.coef_)
  print("Lambda dictionary:", {feature:coeffient for feature,coeffient in zip(feature_columns, model.coef_)})

  runtime = time.time() - start_time  
  print("Finished generating data in %.2f seconds." %(runtime))
