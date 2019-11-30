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

  def __init__(self, name, exploration_c=100, depth=3, discount_factor=0.9, num_iterations=1000):
    self.name = name
    self.exploration_c = exploration_c
    self.depth = depth
    self.discount_factor = discount_factor
    self.num_iterations = num_iterations

  def get_name(self):
    return self.name

  def get_action(self, agent_player, game):
    opp_player = Player.PLAYER_1 if agent_player == Player.PLAYER_2 else Player.PLAYER_1
    next_player = {Player.PLAYER_1 : Player.PLAYER_2, Player.PLAYER_2 : Player.PLAYER_1}
    Q = {agent_player: {}, opp_player: {}}
    N = {agent_player: {}, opp_player: {}}
    T = {agent_player: [], opp_player: []}

    def eval_four_helper(series, my_player, other_player):
      if series.count(my_player) == 4:
        return 1
      if series.count(my_player) == 3 and series.count(Player.NONE) == 1:
        return 0.00008
      if series.count(my_player) == 2 and series.count(Player.NONE) == 2:
        return 0.00006
      if series.count(other_player) == 2 and series.count(Player.NONE) == 2:
        return -0.00007
      if series.count(other_player) == 3 and series.count(Player.NONE) == 1:
        return -0.00009
      if series.count(other_player) == 4:
        return -1
      return 0

    def heuristic_reward(game, my_player, other_player):
      score = 0
      for row in range(game.NUM_ROWS):
        for col in range(game.NUM_COLS):
          if col + 3 < game.NUM_COLS:
            series = [game.board[row][col+i] for i in range(4)]
            score += eval_four_helper(series, my_player, other_player)
          if row + 3 < game.NUM_ROWS:
            series = [game.board[row+i][col] for i in range(4)]
            score += eval_four_helper(series, my_player, other_player)
          if row + 3 < game.NUM_ROWS and col + 3 < game.NUM_COLS:
            series = [game.board[row+i][col+i] for i in range(4)]
            score += eval_four_helper(series, my_player, other_player)
          if row + 3 < game.NUM_ROWS and col - 3 >= 0:
            series = [game.board[row+i][col-i] for i in range(4)]
            score += eval_four_helper(series, my_player, other_player)
      return score

    def random_policy(game):
      valid_actions = [action for action in range(game.NUM_COLS) if game.valid_action(action)]
      return random.choice(valid_actions)

    def random_rollout(curr_player, my_player, other_player, game, depth):
      if depth == 0:
        return 0

      action = random_policy(game)
      next_game = game.add_piece(curr_player, action)

      # terminal state
      if next_game.check_draw():
        #print("RANDOM ROLLOUT - FOUND DRAW!!!") #+++
        next_game.print_board()
        return 0
      if next_game.check_win(my_player):
        #print("RANDOM ROLLOUT - FOUND PLAYER WIN FOR %s!!!" %(my_player)) #+++
        next_game.print_board()
        return 1
      if next_game.check_win(other_player):
        #print("RANDOM ROLLOUT - FOUND OPPONENT WIN FOR %s!!!" %(my_player)) #+++
        next_game.print_board()
        return -1
      reward = 0
      #reward = heuristic_reward(next_game, my_player, other_player)
      return reward + self.discount_factor * rollout(next_player[curr_player], my_player, other_player, next_game, depth-1 if curr_player == opp_player else depth)

    def rollout(curr_player, my_player, other_player, game, depth):
      return random_rollout(curr_player, my_player, other_player, game, depth)

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
        return rollout(curr_player, curr_player, next_player[curr_player], game, depth)

      best_Q = float("-inf")
      best_action = None
      valid_actions = random.sample(Q[curr_player][state].keys(), len(Q[curr_player][state].keys()))
      for action in valid_actions:
        if N[curr_player][state][action] == 0:
          curr_Q = float("inf")
        else:
          curr_Q = Q[curr_player][state][action] + self.exploration_c * math.sqrt(math.log(sum(N[curr_player][state])) / N[curr_player][state][action])
        if best_Q < curr_Q:
          best_Q = curr_Q
          best_action = action

      next_game = game.add_piece(curr_player, best_action)
      # terminal state
      if next_game.check_draw():
        #print("SIMULATION - FOUND DRAW!!!") #+++
        next_game.print_board()
        q = 0
      elif next_game.check_win(curr_player):
        #print("SIMULATION - FOUND WIN FOR %s!!!" %(curr_player)) #+++
        next_game.print_board()
        q = 1
      elif next_game.check_win(next_player[curr_player]):
        #print("SIMULATION - FOUND OPPONENT WIN FOR !!!" %(curr_player)) #+++
        next_game.print_board()
        q = -1
      else:
        reward = 0
        #reward = heuristic_reward(next_game, curr_player, next_player[curr_player])
        q = reward - self.discount_factor * simulate(next_player[curr_player], next_game, depth-1 if curr_player == opp_player else depth)
      N[curr_player][state][best_action] += 1
      Q[curr_player][state][best_action] += (q - Q[curr_player][state][best_action]) / N[curr_player][state][best_action] # Running average
      return q

    # Update Q and N
    for i in range(self.num_iterations): #main loop
      simulate(agent_player, game, self.depth)
      print("updated agent Q-function:")
      print(Q[agent_player][game.serialize_board()])
      #print("updated opp Q-function:")
      #print(Q[opp_player][game.serialize_board()])
    ##########
    #print("final opp Q-function:") #+++
    #for key, val in Q[opp_player].items():
    #  print(key,":",val)
    #print("final agent Q-function:") #+++
    #for key, val in Q[agent_player].items():
    #  print(key,":",val)
    print("Board analyzed:") #+++
    game.print_board()
    print("=====================")
    ############

    # Choosing best action
    state = game.serialize_board()
    best_Q = float("-inf")
    best_action = None
    for action in random.sample(Q[agent_player][state].keys(), len(Q[agent_player][state])):
      if Q[agent_player][state][action] > best_Q:
        best_Q = Q[agent_player][state][action]
        best_action = action
    return best_action
