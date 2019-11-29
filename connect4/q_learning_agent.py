from basis_function import basis_vector
from old_basis_function import basis_vector_old
from connect4 import Connect4Board
from connect4 import GameState
from connect4 import Player

import numpy as np
import random

class QLearningAgent:
  name = None
  theta_online = None
  theta_offline = None

  def __init__(self, name):
    self.name = name
    self.theta_online = {
      "opponent_3_out_of_5": 2234.1711460280335,
      "opponent_num_possible_wins_in_col_5": 3155.285981413819,
      "player_num_consecutive_possible_wins_in_col_0": 815.4259299265982,
      "player_num_consecutive_possible_wins_in_col_1": 905.5848444084356,
      "player_num_consecutive_possible_wins_in_col_3": 675.512658663018,
      "player_num_possible_wins_in_col_4": 3948.85725781515,
      "opponent_num_possible_wins_in_col_4": 2698.038784009236,
      "player_num_possible_wins_in_col_0": 5205.287443580301,
      "player_num_possible_wins_in_col_5": 5468.343620164691,
      "opponent_num_consecutive_possible_wins_in_col_4": -451.4644265691778,
      "opponent_num_consecutive_possible_wins_in_col_1": -648.1082800283297,
      "player_num_possible_wins_in_col_3": 4027.9377536470606,
      "player_3_out_of_4": 65398.318299813924,
      "opponent_num_consecutive_possible_wins_in_col_5": -691.0393096329848,
      "opponent_3_out_of_4": 62895.38931085311,
      "opponent_num_possible_wins_in_col_3": 2661.7492898161704,
      "player_num_possible_wins_in_col_2": 3665.1677576771995,
      "opponent_num_consecutive_possible_wins_in_col_6": -751.0132552897926,
      "opponent_num_possible_wins_in_col_0": 3136.365397048726,
      "player_num_consecutive_possible_wins_in_col_6": 769.6534195938785,
      "player_2_out_of_4": 480616.0588197917,
      "player_num_possible_wins_in_col_1": 5315.977325840604,
      "opponent_num_possible_wins_in_col_1": 3062.093480092066,
      "opponent_num_consecutive_possible_wins_in_col_0": -713.9369469180211,
      "player_num_consecutive_possible_wins_in_col_5": 912.1983397127533,
      "player_num_possible_wins_in_col_6": 5046.887076701871,
      "opponent_num_possible_wins_in_col_2": 2604.190407159641,
      "opponent_win": -3271.7707821310923,
      "opponent_2_out_of_4": 477959.5404155857,
      "opponent_num_consecutive_possible_wins_in_col_2": -499.92223054723047,
      "player_win": 3348.189361741802,
      "player_num_consecutive_possible_wins_in_col_4": 549.2087537153246,
      "player_3_out_of_5": 1985.9900207949254,
      "player_num_consecutive_possible_wins_in_col_2": 493.44939806925066,
      "opponent_num_consecutive_possible_wins_in_col_3": -257.43127302038204,
      "opponent_num_possible_wins_in_col_6": 2915.0167091017843
    }
    self.theta_offline = {
      "opponent_num_consecutive_possible_wins_in_col_6": 0.11538169490180997,
      "player_3_out_of_5": -0.11513536361430975,
      "player_num_possible_wins_in_col_1": 0.19189436275041022,
      "opponent_num_possible_wins_in_col_4": 0.002697586608265343,
      "player_num_consecutive_possible_wins_in_col_0": -0.12940912150684522,
      "opponent_num_consecutive_possible_wins_in_col_5": 0.013061638161713436,
      "opponent_3_out_of_4": -0.11511284899281453,
      "opponent_num_possible_wins_in_col_2": -0.007134205790131655,
      "player_2_out_of_4": 0.022754246037460145,
      "opponent_num_consecutive_possible_wins_in_col_1": 0.1767520310489181,
      "player_num_possible_wins_in_col_6": 0.11749176295636639,
      "player_num_consecutive_possible_wins_in_col_2": -0.09382556888607123,
      "opponent_2_out_of_4": -0.02275424603746038,
      "opponent_num_possible_wins_in_col_6": -0.11749176295637079,
      "player_num_possible_wins_in_col_4": -0.002697586608266483,
      "opponent_num_consecutive_possible_wins_in_col_3": 0.03394059335649884,
      "player_num_consecutive_possible_wins_in_col_3": -0.03394059335649879,
      "player_3_out_of_4": 0.11511284899281102,
      "player_num_possible_wins_in_col_3": 0.041945887470128645,
      "player_num_possible_wins_in_col_5": 0.051824595725930435,
      "player_win": 0.6791091400977652,
      "player_num_consecutive_possible_wins_in_col_1": -0.17675203104892734,
      "opponent_num_consecutive_possible_wins_in_col_0": 0.12940912150684775,
      "opponent_num_possible_wins_in_col_1": -0.19189436275039826,
      "player_num_consecutive_possible_wins_in_col_6": -0.1153816949018001,
      "player_num_consecutive_possible_wins_in_col_4": -0.018661807070035296,
      "player_num_possible_wins_in_col_0": 0.14220387674439686,
      "opponent_num_consecutive_possible_wins_in_col_4": 0.018661807070038224,
      "opponent_3_out_of_5": 0.11513536361431677,
      "opponent_num_possible_wins_in_col_0": -0.14220387674439983,
      "opponent_num_possible_wins_in_col_5": -0.051824595725932524,
      "opponent_win": -0.6791091400977732,
      "opponent_num_consecutive_possible_wins_in_col_2": 0.09382556888607341,
      "player_num_consecutive_possible_wins_in_col_5": -0.013061638161708305,
      "player_num_possible_wins_in_col_2": 0.007134205790132033,
      "opponent_num_possible_wins_in_col_3": -0.041945887470128375
  }

  def get_name(self):
    return self.name

  def get_action(self, player, game):
    def q_value(game, player):
      basis = basis_vector_old(game, player)
      q = 0
      for key in basis.keys():
        #q += self.theta_online[key]*basis[key] #real bad
        q += self.theta_offline[key]*basis[key] #seems like it works
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
