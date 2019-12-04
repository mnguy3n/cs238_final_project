from basis_function import basis_vector
from connect4 import Connect4Board
from connect4 import GameState
from connect4 import Player

import collections
import math
import numpy as np
import random

class MCTSAgent:
  name = None
  exploration_c = None
  depth = None
  discount_factor = None
  num_iterations = None
  theta = None

  def __init__(self, name, exploration_c=0.9, depth=5, discount_factor=0.9, num_iterations=500):
    self.name = name
    self.exploration_c = exploration_c
    self.depth = depth
    self.discount_factor = discount_factor
    self.num_iterations = num_iterations
    self.theta = {
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

  def get_action(self, agent_player, game):
    next_player = {Player.PLAYER_1 : Player.PLAYER_2, Player.PLAYER_2 : Player.PLAYER_1}
    opp_player = Player.PLAYER_1 if agent_player == Player.PLAYER_2 else Player.PLAYER_2
    Q = {agent_player: {}, opp_player: {}}
    N = {agent_player: {}, opp_player: {}}
    T = {agent_player: [], opp_player: []}

    def rollout(curr_player, game):
      basis = basis_vector(game, curr_player)
      q = 0
      for key in basis.keys():
        q += self.theta[key] * basis[key]
      return q

    def simulate(curr_player, game, depth):
      if depth == 0:
        return 0

      state = game.serialize_board()
      if not state in T[curr_player]:
        # Expansion
        Q[curr_player][state] = {}
        N[curr_player][state] = {}
        valid_actions = [action for action in range(game.NUM_COLS) if game.valid_action(action)]
        for action in valid_actions:
          Q[curr_player][state][action] = 0
          N[curr_player][state][action] = 0
        T[curr_player].append(state) # Add leaf node to tree
        return rollout(curr_player, game)

      best_Q = float("-inf")
      best_action = None
      valid_actions = random.sample(Q[curr_player][state].keys(), len(Q[curr_player][state].keys()))
      for action in valid_actions:
        if N[curr_player][state][action] == 0 or sum(N[curr_player][state]) == 0:
          curr_Q = float("inf")
        else:
          curr_Q = Q[curr_player][state][action] + self.exploration_c * math.sqrt(math.log(sum(N[curr_player][state])) / N[curr_player][state][action])
        if best_Q < curr_Q:
          best_Q = curr_Q
          best_action = action

      next_game = game.add_piece(curr_player, best_action)
      # terminal state
      if next_game.check_draw():
        q = 0
      elif next_game.check_win(curr_player):
        q = 1
      elif next_game.check_win(next_player[curr_player]):
        q = -1
      else:
        reward = 0
        q = reward - self.discount_factor * simulate(next_player[curr_player], next_game, depth-1 if curr_player == opp_player else depth)
      N[curr_player][state][best_action] += 1
      Q[curr_player][state][best_action] += (q - Q[curr_player][state][best_action]) / N[curr_player][state][best_action] # Running average
      return q

    # Update Q and N
    for i in range(self.num_iterations): #main loop
      simulate(agent_player, game, self.depth)

    # Choosing best action
    state = game.serialize_board()
    best_Q = float("-inf")
    best_action = None
    for action in random.sample(Q[agent_player][state].keys(), len(Q[agent_player][state])):
      if Q[agent_player][state][action] > best_Q:
        best_Q = Q[agent_player][state][action]
        best_action = action
    return best_action
