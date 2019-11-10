from connect4 import Connect4Board
from connect4 import GameState
from connect4 import Player

def test_win_condition():
  # check vertical
  for row in range(6-3):
    for col in range(7):
      game = Connect4Board()
      for i in range(row, row+4):
        game.board[i][col] = Player.PLAYER_1
      if not game.check_win(Player.PLAYER_1):
        return False, "Failed vertical win condition for player 1", game
      game = Connect4Board()
      for i in range(row, row+4):
        game.board[i][col] = Player.PLAYER_2
      if not game.check_win(Player.PLAYER_2):
        return False, "Failed vertical win condition for player 2", game


  # check horizontal
  for row in range(6):
    for col in range(7-3):
      game = Connect4Board()
      for i in range(col, col+4):
        game.board[row][i] = Player.PLAYER_1
      if not game.check_win(Player.PLAYER_1):
        return False, "Failed horizontal win condition for player 1", game
      game = Connect4Board()
      for i in range(col, col+4):
        game.board[row][i] = Player.PLAYER_2
      if not game.check_win(Player.PLAYER_2):
        return False, "Failed horizontal win condition for player 2", game

  # check diagonal
  for row in range(6-3):
    for col in range(7-3):
      game = Connect4Board()
      for i in range(4):
        game.board[row+i][col+i] = Player.PLAYER_1
      if not game.check_win(Player.PLAYER_1):
        return False, "Failed diagonal / win condition for player 1", game
      game = Connect4Board()
      for i in range(4):
        game.board[row+i][col+i] = Player.PLAYER_2
      if not game.check_win(Player.PLAYER_2):
        return False, "Failed diagonal / win condition for player 2", game
  for row in range(6-3):
    for col in range(3,7):
      game = Connect4Board()
      for i in range(4):
        game.board[row+i][col-i] = Player.PLAYER_1
      if not game.check_win(Player.PLAYER_1):
        return False, "Failed diagonal \ win condition for player 1:", game
      game = Connect4Board()
      for i in range(4):
        game.board[row+i][col-i] = Player.PLAYER_2
      if not game.check_win(Player.PLAYER_2):
        return False, "Failed diagonal \ win condition for player 2", game

  return True, None, None

if __name__ == "__main__":
  all_tests_pass = True
  test_win_condition_pass, error, game = test_win_condition()
  if not test_win_condition_pass:
    print("Failed win condition test! Error:", error)
    game.print_board()
    all_tests_pass = False
  if all_tests_pass:
    print("All tests pass!")
