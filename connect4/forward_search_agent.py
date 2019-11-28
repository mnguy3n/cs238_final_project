from connect4 import Connect4Board
from connect4 import GameState
from connect4 import Player

import numpy as np
import random

class ForwardSearchAgent:
  name = None
  depth = None
  discount_factor = None
  agent_two_val = None
  agent_three_val = None
  opp_two_val = None
  opp_three_val = None

  def __init__(self, name, depth=2, discount_factor=0.9, agent_two_val=60, agent_three_val=80, opp_two_val=-70, opp_three_val=-90):
    self.name = name
    self.depth = depth
    self.discount_factor = discount_factor
    self.agent_two_val = agent_two_val
    self.agent_three_val = agent_three_val
    self.opp_two_val = opp_two_val
    self.opp_three_val = opp_three_val


  def get_name(self):
    return self.name

  def get_action(self, player, game):
    next_player = {Player.PLAYER_1 : Player.PLAYER_2, Player.PLAYER_2 : Player.PLAYER_1}
    opp_player = Player.PLAYER_1 if player == Player.PLAYER_2 else Player.PLAYER_2

    def uniform_random_opp_actions(game):
      valid_actions = [action for action in range(game.NUM_COLS) if game.valid_action(action)]
      transition_prob = 1.0 / len(valid_actions)
      return {action: transition_prob for action in valid_actions}

    def forward_search(game, curr_player, curr_depth):
      # end of depth
      if curr_depth == 0:
        return None, 0

      # player = agent
      best_action = None
      best_val = float("-inf") if curr_player == player else float("inf")
      valid_actions = [action for action in range(game.NUM_COLS) if game.valid_action(action)]
      #random.shuffle(valid_actions)
      for action in valid_actions:
        val = val_function(game)
        after_move_game = game.add_piece(curr_player, action)
        if after_move_game.check_draw():
          val = 0
        elif after_move_game.check_win(player):
          val = 100000000
        elif after_move_game.check_win(opp_player):
          val = -100000000
        else:
          _, next_val = forward_search(after_move_game, next_player[curr_player], curr_depth-1 if curr_player == opp_player else curr_depth)
          val += self.discount_factor * next_val

        if curr_player == player and val > best_val or curr_player == opp_player and val < best_val:
          best_val = val
          best_action = action
        ####### FOR DEBUGGING
        #if curr_depth == self.depth and curr_player == player:
        #  print("For player %s, depth %d, and action %d, value was %.4f" %(str(curr_player), curr_depth, action, val))
        #########
      return best_action, best_val

    def val_function(game):
      score = 0
      for row in range(game.NUM_ROWS):
        for col in range(game.NUM_COLS):
          if col + 3 < game.NUM_COLS:
            series = [game.board[row][col+i] for i in range(4)]
            score += val_four_helper(series, player, opp_player)
          if row + 3 < game.NUM_ROWS:
            series = [game.board[row+i][col] for i in range(4)]
            score += val_four_helper(series, player, opp_player)
          if row + 3 < game.NUM_ROWS and col + 3 < game.NUM_COLS:
            series = [game.board[row+i][col+i] for i in range(4)]
            score += val_four_helper(series, player, opp_player)
          if row + 3 < game.NUM_ROWS and col - 3 >= 0:
            series = [game.board[row+i][col-i] for i in range(4)]
            score += val_four_helper(series, player, opp_player)
      return score

    def val_four_helper(series, agent_player, opp_player):
      if series.count(agent_player) == 4:
        return 100000
      if series.count(agent_player) == 3 and series.count(Player.NONE) == 1:
        return self.agent_three_val
      if series.count(agent_player) == 2 and series.count(Player.NONE) == 2:
        return self.agent_two_val
      if series.count(opp_player) == 2 and series.count(Player.NONE) == 2:
        return self.opp_two_val
      if series.count(opp_player) == 3 and series.count(Player.NONE) == 1:
        return self.opp_three_val
      if series.count(opp_player) == 4:
        return -100000
      return 0

    action,_ = forward_search(game, player, self.depth)
    return action
