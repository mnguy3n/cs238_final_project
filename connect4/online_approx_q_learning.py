from basis_function import basis_vector
from connect4 import Connect4Board
from connect4 import GameState
from connect4 import Player

import random
import sys
import time

def calculate_q(theta, basis):
  q = 0
  for key in basis.keys():
    q += theta[key]*basis[key]
  return q

def epsilon_greedy(game, curr_player, theta, epsilon = 0.1):
  valid_actions = [action for action in range(game.NUM_COLS) if game.valid_action(action)]
  random.shuffle(valid_actions)
  if random.random() <= epsilon:
    return random.choice(valid_actions)

  best_val = float("-inf")
  best_action = None
  for action in valid_actions:
    next_game = game.add_piece(curr_player, action)
    basis = basis_vector(next_game, curr_player)
    val = calculate_q(theta, basis)
    if val > best_val:
      best_val = val
      best_action = action
  return best_action

def training(num_iterations, discount_factor=0.8):
  next_player = {Player.PLAYER_1: Player.PLAYER_2, Player.PLAYER_2: Player.PLAYER_1}

  # initialize theta
  theta = {
    "player_2_out_of_4": 0,
    "opponent_2_out_of_4": 0,
    "player_3_out_of_4": 0.01,
    "opponent_3_out_of_4": -0.01,
    "player_3_out_of_5": 0.05,
    "opponent_3_out_of_5": -0.05,
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
    "player_num_consecutive_possible_wins_in_col_0": 0.05,
    "player_num_consecutive_possible_wins_in_col_1": 0.05,
    "player_num_consecutive_possible_wins_in_col_2": 0.05,
    "player_num_consecutive_possible_wins_in_col_3": 0.05,
    "player_num_consecutive_possible_wins_in_col_4": 0.05,
    "player_num_consecutive_possible_wins_in_col_5": 0.05,
    "player_num_consecutive_possible_wins_in_col_6": 0.05,
    "opponent_num_consecutive_possible_wins_in_col_0": -0.05,
    "opponent_num_consecutive_possible_wins_in_col_1": -0.05,
    "opponent_num_consecutive_possible_wins_in_col_2": -0.05,
    "opponent_num_consecutive_possible_wins_in_col_3": -0.05,
    "opponent_num_consecutive_possible_wins_in_col_4": -0.05,
    "opponent_num_consecutive_possible_wins_in_col_5": -0.05,
    "opponent_num_consecutive_possible_wins_in_col_6": -0.05,
    "player_win": 10,
    "opponent_win": -10
  }
  N = {}

  # loop through games
  for i in range(num_iterations):
    print("Iteration %d!" %(i+1))
    game = Connect4Board()
    agent_player = Player.PLAYER_1 if i % 2 == 0 else Player.PLAYER_2
    opp_player = Player.PLAYER_1 if i % 2 != 0 else Player.PLAYER_2
    curr_player = Player.PLAYER_1
    game_end = False
    winner = None

    while True:
      # Choose action based on theta^T * basis + some exploration
      action = epsilon_greedy(game, curr_player, theta)

      # Observe new next state and reward
      game = game.add_piece(curr_player, action)
      reward = 0
      if game.check_draw():
        reward = 0
        game_end = True
      elif game.check_win(agent_player):
        reward = 1
        game_end = True
        winner = agent_player
      elif game.check_win(opp_player):
        reward = -1
        game_end = True
        winner = opp_player

      # Find the action that maximizes q for the next player
      valid_next_actions = [action for action in range(game.NUM_COLS) if game.valid_action(action)]
      next_q = float("-inf")
      next_action = None
      for action in valid_next_actions:
        temp_basis = basis_vector(game.add_piece(next_player[curr_player], action), next_player[curr_player])
        val = calculate_q(theta, temp_basis)
        if val > next_q:
          next_q = val
          next_action = action

      # Update Theta
      board_string = game.serialize_board()
      if board_string not in N:
        N[board_string] = 1
      else:
        N[board_string] += 1
      alpha = 1.0 / N[board_string]
      basis = basis_vector(game, agent_player)
      if len(valid_next_actions) != 0 and not game_end:
        # Using reward clipping to prevent exploding Q-values
        coefficient = alpha * max(1.0, min(-1.0, (reward \
            + discount_factor*calculate_q(theta, basis_vector(game.add_piece(next_player[curr_player], next_action), agent_player)) \
            - calculate_q(theta, basis))))
      else:
        coefficient = alpha * reward
      for key in theta.keys():
        theta[key] += coefficient * basis[key]
 
      if game_end:
        if winner == None:
          print("DRAW!")
        elif winner == agent_player:
          print("WON!")
        elif winner == opp_player:
          print("LOST!")
        break
      curr_player = next_player[curr_player]
  return theta

if __name__ == "__main__":
  num_iterations = 1
  if len(sys.argv) == 2:
    num_iterations = int(sys.argv[1])

  print("Starting games!")
  start_time = time.time()
  theta = training(num_iterations)
  print("Theta:", theta)
  runtime = time.time() - start_time
  print("Finished generating data in %.2f seconds." %(runtime))
