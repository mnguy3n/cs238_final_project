from enum import Enum
import copy

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

  def valid_action(self, col):
    return col >= 0 and col < self.NUM_COLS and self.col_height[col] < self.NUM_ROWS

  def add_piece(self, player, col):
    if self.col_height[col] == self.NUM_ROWS or col < 0 or col >= self.NUM_COLS:
      raise Exception("Invalid move:", col)

    updated_board = Connect4Board()
    updated_board.board = copy.deepcopy(self.board)
    updated_board.col_height = copy.deepcopy(self.col_height)
    updated_board.board[updated_board.col_height[col]][col] = player
    updated_board.col_height[col] += 1
    return updated_board

  def check_win(self, player):
    for row in range(self.NUM_ROWS):
      for col in range(self.NUM_COLS):
        if self.board[row][col] != player:
          continue

        if col + 3 < self.NUM_COLS and self.board[row][col+1] == player and self.board[row][col+2] == player and self.board[row][col+3] == player or \
           row + 3 < self.NUM_ROWS and self.board[row+1][col] == player and self.board[row+2][col] == player and self.board[row+3][col] == player or \
           row + 3 < self.NUM_ROWS and col + 3 < self.NUM_COLS and self.board[row+1][col+1] == player and self.board[row+2][col+2] == player and self.board[row+3][col+3] == player or \
           row + 3 < self.NUM_ROWS and col - 3 >= 0 and self.board[row+1][col-1] == player and self.board[row+2][col-2] == player and self.board[row+3][col-3] == player:
          return True
    return False

  def check_draw(self):
    return sum(self.col_height) == self.NUM_ROWS * self.NUM_COLS

  def check_game_state(self, player):
    if self.check_draw():
      return GameState.DRAW
    if self.check_win(player):
      if player == Player.PLAYER_1:
        return GameState.PLAYER_1_WIN
      return GameState.PLAYER_2_WIN
    return GameState.INPROGRESS
