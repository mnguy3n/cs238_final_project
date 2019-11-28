from connect4 import Connect4Board
from connect4 import GameState
from connect4 import Player
from basis_function import basis_vector
from minimax_agent import MinimaxAgent

import pandas as pd
import random
import sys
import time

def choose_action(player_agent, player, game):
  if random.randint(1,10) == 1:
    valid_actions = [action for action in range(game.NUM_COLS) if game.valid_action(action)]
    return random.choice(valid_actions)
  return player_agent.get_action(player, game)

def episode_to_data(player_1_episode, player_2_episode, winner, data_columns, discount_factor=0.9):
  data = pd.DataFrame(columns=data_columns)
  num_actions = len(player_1_episode)
  winning_reward = [discount_factor ** ((num_actions-1)-i) * 1 for i in range(num_actions)]
  losing_reward = [discount_factor ** ((num_actions-1)-i) * -1 for i in range(num_actions)]
  if winner == Player.PLAYER_1:
    for i in range(num_actions):
      player_1_episode[i]["reward"] = winning_reward[i]
      player_2_episode[i]["reward"] = losing_reward[i]
      player_1_episode[i]["only_end_reward"] = 0
      player_2_episode[i]["only_end_reward"] = 0
    player_1_episode[num_actions-1]["only_end_reward"] = 1
    player_2_episode[num_actions-1]["only_end_reward"] = -1
  elif winner == Player.PLAYER_2:
    for i in range(num_actions):
      player_1_episode[i]["reward"] = losing_reward[i]
      player_2_episode[i]["reward"] = winning_reward[i]
      player_1_episode[i]["only_end_reward"] = 0
      player_2_episode[i]["only_end_reward"] = 0
    player_1_episode[num_actions-1]["only_end_reward"] = -1
    player_2_episode[num_actions-1]["only_end_reward"] = 1
  else:
    raise Exception("Invalid winner")

  for i in range(num_actions):
    data = data.append(player_1_episode[i], ignore_index=True)
    data = data.append(player_2_episode[i], ignore_index=True)
    #print("Size of dataframe:", data.size)
    #print("Shape of dataframe:", data.shape)
  return data

def export_data(data, filename="connect_4_episode_data.csv"):
  print("Size of output:", data.size)
  print("Shape of output:", data.shape)
  data.to_csv(filename, header=True)

def play_games(num_games):
  player_1_agent = MinimaxAgent("MinimaxAgent1")
  player_2_agent = MinimaxAgent("MinimaxAgent2")
  next_player = {Player.PLAYER_1: Player.PLAYER_2, Player.PLAYER_2: Player.PLAYER_1}
  player_map = {Player.PLAYER_1 : player_1_agent, Player.PLAYER_2 : player_2_agent}

  data_columns = sorted(list(basis_vector(Connect4Board(), Player.PLAYER_1).keys()))
  data_columns.append("reward")
  data_columns.append("only_end_reward")
  data = pd.DataFrame(columns=data_columns)

  # loop through games
  for i in range(num_games):
    print("Game %d!" %(i))
    game = Connect4Board()
    curr_player = Player.PLAYER_1    
    player_1_episode = [basis_vector(game, Player.PLAYER_1)]
    player_2_episode = [basis_vector(game, Player.PLAYER_2)]

    winner = None
    while True:
      game = game.add_piece(curr_player, choose_action(player_map[curr_player], curr_player, game))
      player_1_episode.append(basis_vector(game, Player.PLAYER_1))
      player_2_episode.append(basis_vector(game, Player.PLAYER_2))

      game_state = game.check_game_state(curr_player)
      if game_state == GameState.DRAW:
        print("DRAW")
        break
      if game_state == GameState.PLAYER_1_WIN:
        winner = Player.PLAYER_1
        print("PLAYER 1 WON!")
        break
      if game_state == GameState.PLAYER_2_WIN:
        winner = Player.PLAYER_2
        print("PLAYER 2 WON!")
        break
      curr_player = next_player[curr_player]

    if winner != None:
      episode_data = episode_to_data(player_1_episode, player_2_episode, winner, data_columns)
      #print("Size of episode data:", episode_data.size)
      #print("Shape of episode data:", episode_data.shape)
      data = data.append(episode_data)
      #print("Size of data:", data.size)
      #print("Shape of data:", data.shape)
  export_data(data)

if __name__ == "__main__":
  num_games = 1
  if len(sys.argv) == 2:
    num_games = int(sys.argv[1])

  print("Starting games!")
  start_time = time.time()
  play_games(num_games)
  runtime = time.time() - start_time
  print("Finished generating data in %.2f seconds." %(runtime))
