from connect4 import Connect4Board
from connect4 import GameState
from connect4 import Player
from human_agent import HumanAgent
from minimax_agent import MinimaxAgent

import sys

def run_game(print_board=False):
  game = Connect4Board()
  next_player = {Player.PLAYER_1: Player.PLAYER_2, Player.PLAYER_2: Player.PLAYER_1}
  curr_player = Player.PLAYER_1

  #player_1_agent = HumanAgent("The Human")
  player_1_agent = MinimaxAgent("The replacement human")
  #player_2_agent = HumanAgent("Player 2")
  player_2_agent = MinimaxAgent("The AI")
  player_map = {Player.PLAYER_1 : player_1_agent, Player.PLAYER_2 : player_2_agent}

  while True:
    if print_board:
      game.print_board()
      print("================================")

    game = game.add_piece(curr_player, player_map[curr_player].get_action(curr_player, game))

    game_state = game.check_game_state(curr_player)
    if game_state == GameState.DRAW:
      print("DRAW!!!")
      game.print_board()
      return
    elif game_state == GameState.PLAYER_1_WIN:
      print("PLAYER 1 WINS!!!")
      game.print_board()
      return
    elif game_state == GameState.PLAYER_2_WIN:
      print("PLAYER 2 WINS!!!")
      game.print_board()
      return
    curr_player = next_player[curr_player]

if __name__ == "__main__":
  print_board = True
  if len(sys.argv) > 1:
    print_board = sys.argv[1]
  run_game(print_board)
