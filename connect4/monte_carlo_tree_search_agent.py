from connect4 import Connect4Board
from connect4 import GameState
from connect4 import Player
from minimax_agent import MinimaxAgent

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

  def __init__(self, name, exploration_c=0.9, depth=3, discount_factor=0.9, num_iterations=500):
    self.name = name
    self.exploration_c = exploration_c
    self.depth = depth
    self.discount_factor = discount_factor
    self.num_iterations = num_iterations

  def get_name(self):
    return self.name

  def get_action(self, player, game):
    next_player = {Player.PLAYER_1 : Player.PLAYER_2, Player.PLAYER_2 : Player.PLAYER_1}
    opp_player = Player.PLAYER_1 if player == Player.PLAYER_2 else Player.PLAYER_1
    Q = {}
    N = {}
    T = []

    def eval_four_helper(series, player_one, player_two):
      if series.count(player_one) == 4:
        return 100000
      if series.count(player_one) == 3 and series.count(Player.NONE) == 1:
        return 20
      if series.count(player_one) == 2 and series.count(Player.NONE) == 2:
        return 5
      if series.count(player_two) == 2 and series.count(Player.NONE) == 2:
        return -2
      if series.count(player_two) == 3 and series.count(Player.NONE) == 1:
        return -10
      if series.count(player_two) == 4:
        return -100000
      return 0

    def heuristic_reward(game, player_one, player_two):
      score = 0
      for row in range(game.NUM_ROWS):
        for col in range(game.NUM_COLS):
          if col + 3 < game.NUM_COLS:
            series = [game.board[row][col+i] for i in range(4)]
            score += eval_four_helper(series, player_one, player_two)
          if row + 3 < game.NUM_ROWS:
            series = [game.board[row+i][col] for i in range(4)]
            score += eval_four_helper(series, player_one, player_two)
          if row + 3 < game.NUM_ROWS and col + 3 < game.NUM_COLS:
            series = [game.board[row+i][col+i] for i in range(4)]
            score += eval_four_helper(series, player_one, player_two)
          if row + 3 < game.NUM_ROWS and col - 3 >= 0:
            series = [game.board[row+i][col-i] for i in range(4)]
            score += eval_four_helper(series, player_one, player_two)
      return score


    def generative_model(game):
      # terminal state after agent move
      if game.check_draw():
        return game, 0, True
      if game.check_win(player):
        return game, 1, True

      # opponent plays move based on best heuristic
      valid_actions = [action for action in range(game.NUM_COLS) if game.valid_action(action)]
      random.shuffle(valid_actions)
      best_val = float("-inf")
      best_action = None
      for action in valid_actions:
        val = heuristic_reward(game.add_piece(opp_player, action), opp_player, player)
        if val > best_val:
          best_val = val
          best_action = action
      next_game = game.add_piece(opp_player, best_action)

      # terminal state after opponent move
      if next_game.check_draw():
        return next_game, 0, True
      if next_game.check_win(opp_player):
        return next_game, -1, True
      return next_game, 0, False

    def random_policy(game):
      valid_actions = [action for action in range(game.NUM_COLS) if game.valid_action(action)]
      return random.choice(valid_actions)

    def rollout(player, game, depth):
      if depth == 0:
        return 0

      action = random_policy(game)
      next_game, reward, game_end = generative_model(game.add_piece(player, action))
      if game_end:
        return reward
      return reward + self.discount_factor * rollout(player, next_game, depth-1)

    def simulate(player, game, depth):
      if depth == 0:
        return 0

      state = game.serialize_board()
      if not state in T:
        # Expansion
        Q[state] = {}
        N[state] = {}
        valid_actions = [action for action in range(game.NUM_COLS) if game.valid_action(action)]
        for action in valid_actions:
          Q[state][action] = 0
          N[state][action] = 0
        T.append(state) # Add leaf node to tree
        return rollout(player, game, depth)

      best_Q = float("-inf")
      best_action = None
      for action in random.sample(Q[state].keys(), len(Q[state])):
        if N[state][action] == 0:
          curr_Q = float("inf")
        else:
          curr_Q = Q[state][action] + self.exploration_c * math.sqrt(math.log(sum(N[state])) / N[state][action])
        if best_Q < curr_Q:
          best_Q = curr_Q
          best_action = action

      next_game, reward, game_end = generative_model(game.add_piece(player, best_action))
      if game_end:
        q = reward
      else:
        q = reward + self.discount_factor * simulate(player, next_game, depth-1)
      N[state][best_action] += 1
      Q[state][best_action] += (q - Q[state][best_action]) / N[state][best_action] # Running average
      return q

    # Update Q and N
    for i in range(self.num_iterations): #main loop
      simulate(player, game, self.depth)
      #print("Q-function:", Q[game.serialize_board()]) #+++

    # Choosing best action
    state = game.serialize_board()
    best_Q = float("-inf")
    best_action = None
    for action in random.sample(Q[state].keys(), len(Q[state])):
      if Q[state][action] > best_Q:
        best_Q = Q[state][action]
        best_action = action
    return best_action
