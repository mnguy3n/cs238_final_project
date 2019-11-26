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

  def __init__(self, name, exploration_c=0.9, depth=3, discount_factor=0.3, num_iterations=30):
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

    def only_reward_on_win(game):
      if game.check_win(player):
        return 100
      if game.check_win(opp_player):
        return -100
      return 0

    def eval_four_helper(series, agent_player, opp_player):
      if series.count(agent_player) == 4:
        return 100000
      if series.count(agent_player) == 3 and series.count(Player.NONE) == 1:
        return 20
      if series.count(agent_player) == 2 and series.count(Player.NONE) == 2:
        return 5
      if series.count(opp_player) == 2 and series.count(Player.NONE) == 2:
        return -2
      if series.count(opp_player) == 3 and series.count(Player.NONE) == 1:
        return -10
      if series.count(opp_player) == 4:
        return -100000
      return 0

    def heuristic_reward(game):
      score = 0
      for row in range(game.NUM_ROWS):
        for col in range(game.NUM_COLS):
          if col + 3 < game.NUM_COLS:
            series = [game.board[row][col+i] for i in range(4)]
            score += eval_four_helper(series, player, opp_player)
          if row + 3 < game.NUM_ROWS:
            series = [game.board[row+i][col] for i in range(4)]
            score += eval_four_helper(series, player, opp_player)
          if row + 3 < game.NUM_ROWS and col + 3 < game.NUM_COLS:
            series = [game.board[row+i][col+i] for i in range(4)]
            score += eval_four_helper(series, player, opp_player)
          if row + 3 < game.NUM_ROWS and col - 3 >= 0:
            series = [game.board[row+i][col-i] for i in range(4)]
            score += eval_four_helper(series, player, opp_player)
      return score

    def reward_from_state(game):
      #return only_reward_on_win(game)
      return heuristic_reward(game)

    def random_model(player, game, action):
      # make agent move
      after_agent_move_game = game.add_piece(player, action)

      # if end state
      if game.check_draw():
        return after_agent_move_game, 0
      if game.check_win(player):
        return after_agent_move_game, 100000000

      # make opponent move
      valid_actions = [action for action in range(game.NUM_COLS) if game.valid_action(action)]
      after_opp_move_game = after_agent_move_game.add_piece(opp_player, random.choice(valid_actions))
      if game.check_win(opp_player):
        return after_opp_move_game, -100000000
      reward = reward_from_state(after_opp_move_game)
      return after_opp_move_game, reward

    def minimax_model(player, game, action):
      # make agent move
      after_agent_move_game = game.add_piece(player, action)

      # make opponent move
      opp_agent = MinimaxAgent("_", depth=2)
      opp_action = opp_agent.get_action(opp_player, after_agent_move_game)
      after_opp_move_game = after_agent_move_game.add_piece(opp_player, opp_action)
      reward = reward_from_state(after_opp_move_game)
      return after_opp_move_game, reward

    def generative_model(player, game, action): # makes the opponent move so next state is agent's next turn
      #return random_model(player, game, action)
      return minimax_model(player, game, action)

    def random_policy(player, game):
      valid_actions = [action for action in range(game.NUM_COLS) if game.valid_action(action)]
      return random.choice(valid_actions)

    def rollout_policy(player, game):
      return random_policy(player, game)

    def rollout(player, game, depth):
      if depth == 0:
        return 0
      # terminal state
      if game.check_draw():
        return 0
      if game.check_win(player):
        return 100000
      if game.check_win(opp_player):
        return -100000

      action = rollout_policy(player, game)
      next_game, reward = generative_model(player, game, action)
      return reward + self.discount_factor * rollout(player, next_game, depth-1)

    def simulate(player, game, depth):
      if depth == 0:
        return 0

      # terminal state
      if game.check_draw():
        return 0
      if game.check_win(player):
        return 100000
      if game.check_win(opp_player):
        return -100000


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

      next_game, reward = generative_model(player, game, best_action)
      q = reward + self.discount_factor * simulate(player, next_game, depth-1)
      N[state][action] += 1
      Q[state][action] += (q - Q[state][action]) / N[state][action] # Running average
      return q

    # Update Q and N
    for i in range(self.num_iterations): #main loop
      print("Q before simulation",i,":",Q) #+++
      #print("len(Q) before simulation:", len(Q)) #+++
      simulate(player, game, self.depth)
    print("final Q-function:", Q) #+++

    # Choosing best action
    state = game.serialize_board()
    best_Q = float("-inf")
    best_action = None
    for action in random.sample(Q[state].keys(), len(Q[state])):
      if Q[state][action] > best_Q:
        best_Q = Q[state][action]
        best_action = action
    return best_action
