from enum import Enum

class Player(Enum):
  NONE = 0
  PLAYER_1 = 1
  PLAYER_2 = 2

  def __str__(self):
    conversion = {self.NONE : '-', self.PLAYER_1: 'X', self.PLAYER_2: 'O'}
    return conversion[self]

class GameState(Enum):
  INPROGRESS = 0
  PLAYER_1_WIN = 1
  PLAYER_2_WIN = 2
  DRAW = 3

class Connect4Board:
  NUM_ROWS = 6
  NUM_COLS = 7
  board = None
  col_height = None

  def __init__(self):
    # Left-bottom corner is (0,0)
    self.board = [[Player.NONE for col in range(self.NUM_COLS)] for row in range(self.NUM_ROWS)]
    self.col_height = [0 for _ in range(self.NUM_COLS)]

  def print_board(self):
    for row in reversed(range(self.NUM_ROWS)):
      print(" ".join([str(cell) for cell in self.board[row]]))

  def add_piece(self, player, col):
    if self.col_height[col] == self.NUM_ROWS or col < 0 or col >= self.NUM_COLS:
      return False

    self.board[self.col_height[col]][col] = player
    self.col_height[col] += 1
    return True

  def check_win(self, player):
    for row in range(self.NUM_ROWS):
      for col in range(self.NUM_COLS):
        if self.board[row][col] != player:
          continue

        if col + 3 < self.NUM_COLS and self.board[row][col+1] == player and self.board[row][col+2] == player and self.board[row][col+3] == player or \
           row + 3 < self.NUM_ROWS and self.board[row+1][col] == player and self.board[row+2][col] == player and self.board[row+3][col] == player or \
           row + 3 < self.NUM_ROWS and col + 3 < self.NUM_COLS and self.board[row+1][col+1] == player and self.board[row+2][col+2] == player and self.board[row+3][col+3] == player or \
           row + 3 < self.NUM_ROWS and col - 3 > 0 and self.board[row+1][col-1] == player and self.board[row+2][col-2] == player and self.board[row+3][col-3] == player:
          if row + 3 < self.NUM_ROWS and col + 3 < self.NUM_COLS and self.board[row+1][col+1] == player and self.board[row+2][col+2] == player and self.board[row+3][col+3] == player:
            print("DIAGONAL 1")
          if row + 3 < self.NUM_ROWS and col - 3 > 0 and self.board[row+1][col-1] == player and self.board[row+2][col-2] == player and self.board[row+3][col-3] == player:
            print("DIAGONAL 2")
          return True
    return False

  def check_game_state(self, player):
    if sum(self.col_height) == self.NUM_ROWS * self.NUM_COLS:
      return GameState.DRAW
    if self.check_win(player):
      if player == Player.PLAYER_1:
        return GameState.PLAYER_1_WIN
      return GameState.PLAYER_2_WIN
    return GameState.INPROGRESS
