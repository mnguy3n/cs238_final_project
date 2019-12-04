from basis_function import basis_vector
from basis_function import basis_vector
from connect4 import Connect4Board
from connect4 import GameState
from connect4 import Player

import numpy as np
import random

class QLearningAgent:
  name = None
  use_offline_params = None
  theta_online = None
  theta_offline = None

  def __init__(self, name, use_offline_params = False):
    self.name = name
    self.use_offline_params = use_offline_params
    self.theta_online = {
      "player_num_possible_wins_in_col_3": 8139.292023795227,
      "opponent_3_out_of_5": 2561.271758401387,
      "opponent_num_consecutive_possible_wins_in_col_0": 46.73571428571428,
      "opponent_num_consecutive_possible_wins_in_col_1": 475.7211703185538,
      "player_num_possible_wins_in_col_1": 8777.713527430345,
      "opponent_num_consecutive_possible_wins_in_col_3": 1285.4441669256018,
      "opponent_num_consecutive_possible_wins_in_col_2": 2525.2296120645165,
      "opponent_num_possible_wins_in_col_2": 14622.372716140242,
      "opponent_num_possible_wins_in_col_1": 8791.161880415817,
      "player_num_consecutive_possible_wins_in_col_4": 2545.6364189051965,
      "player_num_consecutive_possible_wins_in_col_1": 419.86472573265627,
      "player_num_consecutive_possible_wins_in_col_0": 12.05,
      "opponent_num_consecutive_possible_wins_in_col_5": 405.6517231182447,
      "opponent_num_possible_wins_in_col_4": 15419.644292408922,
      "player_num_possible_wins_in_col_5": 8716.604582743756,
      "player_num_possible_wins_in_col_6": 4590.894549704054,
      "opponent_num_consecutive_possible_wins_in_col_6": 33.67738095238096,
      "player_num_possible_wins_in_col_0": 4541.45043239415,
      "player_num_consecutive_possible_wins_in_col_5": 389.90075993694416,
      "opponent_num_consecutive_possible_wins_in_col_4": 2637.5778891690657,
      "player_num_consecutive_possible_wins_in_col_3": 1365.0624284672979,
      "player_num_possible_wins_in_col_4": 15563.058470657974,
      "player_3_out_of_5": 2121.6246593828373,
      "player_2_out_of_4": 523419.5014114607,
      "opponent_win": -3428.3721526893664,
      "player_num_consecutive_possible_wins_in_col_2": 2814.3064544596305,
      "opponent_2_out_of_4": 517775.44510431204,
      "opponent_num_possible_wins_in_col_5": 8580.171730788232,
      "opponent_num_possible_wins_in_col_3": 8345.846012715356,
      "player_num_consecutive_possible_wins_in_col_6": 17.133333333333333,
      "opponent_3_out_of_4": 70835.96875824689,
      "player_3_out_of_4": 71692.55094079571,
      "opponent_num_possible_wins_in_col_6": 4344.458997589873,
      "opponent_num_possible_wins_in_col_0": 4355.954989445178,
      "player_win": 3434.4884802072806,
      "player_num_possible_wins_in_col_2": 14947.595671627998
    }
    self.theta_offline = {
      "player_num_possible_wins_in_col_3": 0.14359636760651984,
      "player_num_consecutive_possible_wins_in_col_5": 0.06811800950809425,
      "player_num_consecutive_possible_wins_in_col_4": -0.14309882661794826,
      "opponent_3_out_of_5": 0.04125866449222762,
      "opponent_num_consecutive_possible_wins_in_col_2": -0.07155909478062643,
      "player_3_out_of_5": -0.041258664492228495,
      "opponent_num_possible_wins_in_col_5": -0.12706819548969345,
      "player_num_possible_wins_in_col_1": 0.13821540837263072,
      "opponent_win": -0.729016518959015,
      "opponent_num_consecutive_possible_wins_in_col_3": -0.6886344031521995,
      "opponent_3_out_of_4": -0.0732572816754046,
      "player_num_consecutive_possible_wins_in_col_3": 0.6886344031521776,
      "player_num_consecutive_possible_wins_in_col_2": 0.07155909478062634,
      "player_num_possible_wins_in_col_4": 0.10757042551191198,
      "opponent_num_possible_wins_in_col_1": -0.13821540837263496,
      "opponent_num_possible_wins_in_col_6": -0.1413009532163512,
      "player_num_consecutive_possible_wins_in_col_1": -0.2971123695163536,
      "player_num_consecutive_possible_wins_in_col_0": 0.02840868572463406,
      "opponent_num_consecutive_possible_wins_in_col_5": -0.06811800950809999,
      "player_2_out_of_4": 0.018104917640001928,
      "player_num_possible_wins_in_col_0": 0.10060372086326475,
      "player_num_consecutive_possible_wins_in_col_6": -0.0610932311092139,
      "opponent_num_consecutive_possible_wins_in_col_0": -0.028408685724633287,
      "opponent_num_possible_wins_in_col_2": -0.0587029001387138,
      "player_num_possible_wins_in_col_5": 0.12706819548969484,
      "player_win": 0.7290165189590107,
      "player_3_out_of_4": 0.07325728167540471,
      "player_num_possible_wins_in_col_6": 0.14130095321635477,
      "opponent_num_possible_wins_in_col_3": -0.14359636760651828,
      "opponent_num_consecutive_possible_wins_in_col_6": 0.0610932311092044,
      "opponent_num_possible_wins_in_col_4": -0.10757042551191245,
      "opponent_2_out_of_4": -0.018104917640002962,
      "opponent_num_consecutive_possible_wins_in_col_1": 0.29711236951635067,
      "player_num_possible_wins_in_col_2": 0.05870290013871207,
      "opponent_num_possible_wins_in_col_0": -0.10060372086326627,
      "opponent_num_consecutive_possible_wins_in_col_4": 0.14309882661794818
    }

  def get_name(self):
    return self.name

  def get_action(self, player, game):
    def q_value(game, player):
      basis = basis_vector(game, player)
      q = 0
      for key in basis.keys():
        if self.use_offline_params:
          q += self.theta_offline[key]*basis[key] #seems ok
        else:
          q += self.theta_online[key]*basis[key] #seems not great
      return q

    valid_actions = [action for action in range(game.NUM_COLS) if game.valid_action(action)]
    random.shuffle(valid_actions)
    best_score = float("-inf")
    best_action = None
    for action in valid_actions:
      score = q_value(game.add_piece(player, action), player)
      if score > best_score:
        best_score = score
        best_action = action
    return best_action
