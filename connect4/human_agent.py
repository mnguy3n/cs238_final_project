class HumanAgent:
  name = None
  def __init__(self, name):
    self.name = name

  def get_name(self):
    return self.name

  def get_action(self, player, game):
    action = int(input("Choose a column (0-6): "))
    while not game.valid_action(action):
      action = int(input("Invalid action (0-6): "))
    return action
