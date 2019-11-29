from connect4 import Connect4Board
from connect4 import GameState
from connect4 import Player

import copy
import numpy as np
import random

def basis_vector_old(game, player):
  opp_player = Player.PLAYER_1 if player == Player.PLAYER_2 else Player.PLAYER_2
  basis_vector = {
    "player_2_out_of_4": 0,
    "opponent_2_out_of_4": 0,
    "player_3_out_of_4": 0,
    "opponent_3_out_of_4": 0,
    "player_3_out_of_5": 0,
    "opponent_3_out_of_5": 0,
    "player_num_possible_wins_in_col_0": 0,
    "player_num_possible_wins_in_col_1": 0,
    "player_num_possible_wins_in_col_2": 0,
    "player_num_possible_wins_in_col_3": 0,
    "player_num_possible_wins_in_col_4": 0,
    "player_num_possible_wins_in_col_5": 0,
    "player_num_possible_wins_in_col_6": 0,
    "opponent_num_possible_wins_in_col_0": 0,
    "opponent_num_possible_wins_in_col_1": 0,
    "opponent_num_possible_wins_in_col_2": 0,
    "opponent_num_possible_wins_in_col_3": 0,
    "opponent_num_possible_wins_in_col_4": 0,
    "opponent_num_possible_wins_in_col_5": 0,
    "opponent_num_possible_wins_in_col_6": 0,
    "player_num_consecutive_possible_wins_in_col_0": 0,
    "player_num_consecutive_possible_wins_in_col_1": 0,
    "player_num_consecutive_possible_wins_in_col_2": 0,
    "player_num_consecutive_possible_wins_in_col_3": 0,
    "player_num_consecutive_possible_wins_in_col_4": 0,
    "player_num_consecutive_possible_wins_in_col_5": 0,
    "player_num_consecutive_possible_wins_in_col_6": 0,
    "opponent_num_consecutive_possible_wins_in_col_0": 0,
    "opponent_num_consecutive_possible_wins_in_col_1": 0,
    "opponent_num_consecutive_possible_wins_in_col_2": 0,
    "opponent_num_consecutive_possible_wins_in_col_3": 0,
    "opponent_num_consecutive_possible_wins_in_col_4": 0,
    "opponent_num_consecutive_possible_wins_in_col_5": 0,
    "opponent_num_consecutive_possible_wins_in_col_6": 0,
    "player_win": 0,
    "opponent_win": 0
  }

  for row in range(game.NUM_ROWS):
    for col in range(game.NUM_COLS):
      if col + 3 < game.NUM_COLS:
        series = [game.board[row][col+i] for i in range(4)]
        four_series_helper(series, player, opp_player, basis_vector)
      if row + 3 < game.NUM_ROWS:
        series = [game.board[row+i][col] for i in range(4)]
        four_series_helper(series, player, opp_player, basis_vector)
      if row + 3 < game.NUM_ROWS and col + 3 < game.NUM_COLS:
        series = [game.board[row+i][col+i] for i in range(4)]
        four_series_helper(series, player, opp_player, basis_vector)
      if row + 3 < game.NUM_ROWS and col - 3 >= 0:
        series = [game.board[row+i][col-i] for i in range(4)]
        four_series_helper(series, player, opp_player, basis_vector)

      if col + 4 < game.NUM_COLS:
        series = [game.board[row][col+i] for i in range(5)]
        five_series_helper(series, player, opp_player, basis_vector)
      if row + 4 < game.NUM_ROWS:
        series = [game.board[row+i][col] for i in range(5)]
        five_series_helper(series, player, opp_player, basis_vector)
      if row + 4 < game.NUM_ROWS and col + 4 < game.NUM_COLS:
        series = [game.board[row+i][col+i] for i in range(5)]
        five_series_helper(series, player, opp_player, basis_vector)
      if row + 4 < game.NUM_ROWS and col - 4 >= 0:
        series = [game.board[row+i][col-i] for i in range(5)]
        five_series_helper(series, player, opp_player, basis_vector)

  #if not game.check_win(player) and not game.check_win(opp_player):
  for col in range(game.NUM_COLS):
    for row in range(game.col_height[col], game.NUM_ROWS):
      temp_game = copy.deepcopy(game)
      temp_game.board[row][col] = player
      if check_win_on_column(temp_game, player, col):
        basis_vector["player_num_possible_wins_in_col_" + str(col)] += 1
        if row != game.NUM_ROWS-1:
          temp_game.board[row][col] = Player.NONE
          temp_game.board[row+1][col] = player
          if check_win_on_column(temp_game, player, col):
            basis_vector["player_num_consecutive_possible_wins_in_col_" + str(col)] += 1

      temp_game = copy.deepcopy(game)
      temp_game.board[row][col] = opp_player
      if check_win_on_column(temp_game, opp_player, col):
        basis_vector["opponent_num_possible_wins_in_col_" + str(col)] += 1
        if row != game.NUM_ROWS-1:
          temp_game.board[row][col] = Player.NONE
          temp_game.board[row+1][col] = opp_player
          if check_win_on_column(temp_game, opp_player, col):
            basis_vector["opponent_num_consecutive_possible_wins_in_col_" + str(col)] += 1

  return basis_vector

def four_series_helper(series, agent_player, opp_player, basis_vector):
  if series.count(agent_player) == 4:
    basis_vector["player_win"] += 1
  if series.count(opp_player) == 4:
    basis_vector["opponent_win"] += 1
  if series.count(agent_player) == 3 and series.count(Player.NONE) == 1:
    basis_vector["player_3_out_of_4"] += 1
  if series.count(opp_player) == 3 and series.count(Player.NONE) == 1:
    basis_vector["opponent_3_out_of_4"] += 1
  if series.count(agent_player) == 2 and series.count(Player.NONE) == 2:
    basis_vector["player_2_out_of_4"] += 1
  if series.count(opp_player) == 2 and series.count(Player.NONE) == 2:
    basis_vector["opponent_2_out_of_4"] += 1

def five_series_helper(series, agent_player, opp_player, basis_vector):
  if series[0] == Player.NONE and series[1] == agent_player and series[2] == agent_player and series[3] == agent_player and series[4] == Player.NONE:
    basis_vector["player_3_out_of_5"] += 1
  if series[0] == Player.NONE and series[1] == opp_player and series[2] == opp_player and series[3] == opp_player and series[4] == Player.NONE:
    basis_vector["opponent_3_out_of_5"] += 1

def check_win_on_column(game, player, col):
  for row in range(game.NUM_ROWS):
    if game.board[row][col] != player:
      continue
    if col + 3 < game.NUM_COLS and game.board[row][col+1] == player and game.board[row][col+2] == player and game.board[row][col+3] == player or \
       col - 3 >= 0 and game.board[row][col-1] == player and game.board[row][col-2] == player and game.board[row][col-3] == player or \
       row + 3 < game.NUM_ROWS and game.board[row+1][col] == player and game.board[row+2][col] == player and game.board[row+3][col] == player or \
       row + 3 < game.NUM_ROWS and col + 3 < game.NUM_COLS and game.board[row+1][col+1] == player and game.board[row+2][col+2] == player and game.board[row+3][col+3] == player or \
       row + 3 < game.NUM_ROWS and col - 3 >= 0 and game.board[row+1][col-1] == player and game.board[row+2][col-2] == player and game.board[row+3][col-3] == player:
      return True
  return False
