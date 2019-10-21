from connect4 import Connect4Board
from connect4 import GameState
from connect4 import Player

def run_game():
  game = Connect4Board()
  next_player = {Player.PLAYER_1: Player.PLAYER_2, Player.PLAYER_2: Player.PLAYER_1}
  curr_player = Player.PLAYER_1
  while True:
    game.print_board()
    #print("Choose a column (0-6): ")
    input_col = int(input("Choose a column (0-6): "))
    while not game.add_piece(curr_player, input_col):
      #print("Invalid action (0-6): ")
      input_col = int(input("Invalid action (0-6): "))
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
  run_game()
