from connect4 import Connect4Board
from connect4 import GameState
from connect4 import Player

import numpy as np
import random

class MinimaxAgent:
  player = None
  depth = None
  alpha = None
  beta = None

  def __init__(self, player, depth=3):
    self.player = player
    self.depth = depth

  def get_action(self, game):
    next_player = {Player.PLAYER_1 : Player.PLAYER_2, Player.PLAYER_2 : Player.PLAYER_1}

    def minimax(game, curr_player, curr_depth):
      # terminal state
      if game.check_draw():
        return 0
      if game.check_win(Player.PLAYER_1):
        if self.player == Player.PLAYER_1:
          return 1000000000
        else:
          return -1000000000
      if game.check_win(Player.PLAYER_2):
        if self.player == Player.PLAYER_2:
          return 1000000000
        else:
          return -1000000000

      # end of depth
      if curr_depth == 0:
        return eval_function(game)

      valid_actions = [action for action in range(game.NUM_COLS) if game.valid_action(action)]
      # player = agent
      if curr_player == self.player:
        return max([minimax(game.add_piece(curr_player,action), next_player[curr_player], curr_depth) for action in valid_actions])

      # player = opponent
      else:
        return min([minimax(game.add_piece(curr_player,action), next_player[curr_player], curr_depth-1) for action in valid_actions])

    def alpha_beta_minimax(game, curr_player, curr_depth, alpha, beta):
      # terminal state
      if game.check_draw():
        return 0
      if game.check_win(Player.PLAYER_1):
        if self.player == Player.PLAYER_1:
          return 1000000000
        else:
          return -1000000000
      if game.check_win(Player.PLAYER_2):
        if self.player == Player.PLAYER_2:
          return 1000000000
        else:
          return -1000000000

      # end of depth
      if curr_depth == 0:
        return eval_function(game)

      valid_actions = [action for action in range(game.NUM_COLS) if game.valid_action(action)]
      # player = agent
      if curr_player == self.player:
        #return max([minimax(game.add_piece(curr_player,action), next_player[curr_player], curr_depth) for action in valid_actions]) 
        curr_value = float("-inf")
        for action in valid_actions:
          curr_value = max(curr_value, alpha_beta_minimax(game.add_piece(curr_player, action), next_player[curr_player], curr_depth, alpha, beta))
          alpha = max(alpha, curr_value)
          if alpha >= beta:
            break
        return curr_value

      # player = opponent
      else:
        #return min([minimax(game.add_piece(curr_player,action), next_player[curr_player], curr_depth-1) for action in valid_actions]) 
        curr_value = float("inf")
        for action in valid_actions:
          curr_value = min(curr_value, alpha_beta_minimax(game.add_piece(curr_player, action), next_player[curr_player], curr_depth-1, alpha, beta))
          beta = min(beta, curr_value)
          if alpha >= beta:
            break
        return curr_value

    def eval_function(game):
      opp_player = Player.PLAYER_1 if self.player == Player.PLAYER_2 else Player.PLAYER_2
      score = 0
      for row in range(game.NUM_ROWS):
        for col in range(game.NUM_COLS):
          if col + 3 < game.NUM_COLS:
            series = [game.board[row][col+i] for i in range(4)]
            score += eval_four_helper(series, self.player, opp_player)
          if row + 3 < game.NUM_ROWS:
            series = [game.board[row+i][col] for i in range(4)]
            score += eval_four_helper(series, self.player, opp_player)
          if row + 3 < game.NUM_ROWS and col + 3 < game.NUM_COLS:
            series = [game.board[row+i][col+i] for i in range(4)]
            score += eval_four_helper(series, self.player, opp_player)
          if row + 3 < game.NUM_ROWS and col - 3 >= 0:
            series = [game.board[row+i][col-i] for i in range(4)]
            score += eval_four_helper(series, self.player, opp_player)
      return score

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

    valid_actions = [action for action in range(game.NUM_COLS) if game.valid_action(action)]
    #scores = [minimax(game.add_piece(self.player, action), next_player[self.player], self.depth, float("-inf"), float("inf")) for action in valid_actions]
    scores = [alpha_beta_minimax(game.add_piece(self.player, action), next_player[self.player], self.depth, float("-inf"), float("inf")) for action in valid_actions]
    best_score = max(scores)
    best_indeces = [index for index in range(len(scores)) if scores[index] == best_score]
    return valid_actions[random.choice(best_indeces)]
