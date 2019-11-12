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

  def __init__(self, name, exploration_c=0.9, depth=3, discount_factor=0.9, num_iterations=10):
    self.name = name
    self.exploration_c = exploration_c
    self.depth = depth
    self.discount_factor = discount_factor
    self.num_iterations = num_iterations

  def get_name(self):
    return self.name

  def get_action(self, player, game):
    opp_player = Player.PLAYER_1 if player == Player.PLAYER_2 else Player.PLAYER_1
    Q = {}
    N = {}
    T = []

    def state_string(player, game):
      return str(player) + "|" + game.serialize_board()

    def random_model(player, game, action):
      # make agent move
      after_agent_move_game = game.add_piece(player, action)

      # make opponent move
      valid_actions = [action for action in range(game.NUM_COLS) if game.valid_action(action)]
      after_opp_move_game = after_agent_move_game.add_piece(opp_player, random.choice(valid_actions))

      reward = 0
      if game.check_win(player):
        reward = 100
      elif game.check_win(opp_player):
        reward = -100

      return after_opp_move_game, reward

    def generative_model(player, game, action): # makes the opponent move so next state is agent's next turn
      return random_model(player, game, action)

    def simulate(player, game, depth):
      if depth == 0:
        return 0
      #state = state_string(player, game)
      state = game.serialize_board()
      if not state in T:
        # Expansion
        Q[state] = {}
        N[state] = {}
        valid_actions = [action for action in range(game.NUM_COLS) if game.valid_action(action)]
        for action in valid_actions:
          Q[state][action] = 0
          N[state][action] = 0
        T.append(state)
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

      next_game, reward = generative_model(player, game, best_action)
      q = reward + self.discount_factor * simulate(player, next_game, depth-1)
      N[state][action] += 1
      Q[state][action] += (q - Q[state][action]) / N[state][action]
      return q

    def random_policy(player, game):
      valid_actions = [action for action in range(game.NUM_COLS) if game.valid_action(action)]
      return random.choice(valid_actions)

    def rollout_policy(player, game):
      return random_policy(player, game)

    def rollout(player, game, depth):
      if depth == 0:
        return 0
      action = rollout_policy(player, game)
      next_game, reward = generative_model(player, game, action)
      return reward + self.discount_factor * rollout(player, next_game, depth-1)

    for _ in range(self.num_iterations): #main loop
      simulate(player, game, self.depth)

    state = game.serialize_board()
    #print("STATE:", state) #+++
    #print("Q:", Q) #+++
    #print("N:", N) #+++
    #print("T:", T) #+++

    best_Q = float("-inf")
    best_action = None
    for action in random.sample(Q[state].keys(), len(Q[state])):
      if Q[state][action] > best_Q:
        best_Q = Q[state][action]
        best_action = action
    return best_action
