from connect4 import Connect4Board
from connect4 import GameState
from connect4 import Player
from human_agent import HumanAgent
from minimax_agent import MinimaxAgent

import random
import sys

def instantiate_agents():
  agents = []
  for _ in range(8):
    agent_two_val = random.randint(1,100)
    agent_three_val = random.randint(agent_two_val,100)
    opp_two_val = random.randint(-100,-1)
    opp_three_val = random.randint(-100,opp_two_val)
    agent_name = "_".join(["Agent_depth=3_","AgentThreeVal",str(agent_three_val),"AgentTwoVal",str(agent_two_val),"OppThreeVal",str(opp_three_val),"OppTwoVal",str(opp_two_val)])
    agent_depth_3 = MinimaxAgent(agent_name,depth=3,agent_three_val=agent_three_val,agent_two_val=agent_two_val,opp_three_val=opp_three_val,opp_two_val=opp_two_val)
    agents.append(agent_depth_3)
  return agents

def print_scoreboard(scoreboard):
  for agent, scores_against_others in scoreboard.items():
    total_wins = sum([scores["win"] for opp_agent, scores in scores_against_others.items()])
    total_loss = sum([scores["loss"] for opp_agent, scores in scores_against_others.items()])
    total_draw = sum([scores["draw"] for opp_agent, scores in scores_against_others.items()])
    print("Total score for %s: Total wins=%d, Total losses=%d, Total draws=%d" %(agent, total_wins, total_loss, total_draw))
    for opp_agent, scores in scores_against_others.items():
      print("Scores against %s: Wins=%d, Losses=%d, Draws=%d" %(opp_agent, scores["win"], scores["loss"], scores["draw"]))

def play_games(num_games=8):
  agent_list = instantiate_agents()
  # set up counts for win/loss
  scoreboard = { agent.get_name() : { other_agent.get_name() : {"win":0, "loss":0, "draw":0} for other_agent in agent_list if other_agent.get_name() != agent.get_name()} for agent in agent_list}

  # loop through games
  next_player = {Player.PLAYER_1: Player.PLAYER_2, Player.PLAYER_2: Player.PLAYER_1}
  for player_1_idx in range(len(agent_list)):
    for player_2_idx in range(player_1_idx+1, len(agent_list)):
      player_1_agent = agent_list[player_1_idx]
      player_2_agent = agent_list[player_2_idx]
      print("%s VS %s:" %(player_1_agent.get_name(), player_2_agent.get_name()))
      player_map = {Player.PLAYER_1 : player_1_agent, Player.PLAYER_2 : player_2_agent}

      for i in range(num_games // 2):
        #print("Game %d" %(i+1))
        game = Connect4Board()
        curr_player = Player.PLAYER_1
        winner = None

        while True:
          #game.print_board()
          #print("=======================")
          game = game.add_piece(curr_player, player_map[curr_player].get_action(curr_player, game))

          game_state = game.check_game_state(curr_player)
          if game_state == GameState.DRAW:
            scoreboard[player_1_agent.get_name()][player_2_agent.get_name()]["draw"] += 1
            scoreboard[player_2_agent.get_name()][player_1_agent.get_name()]["draw"] += 1
            winner = None
            break
          if game_state == GameState.PLAYER_1_WIN:
            scoreboard[player_1_agent.get_name()][player_2_agent.get_name()]["win"] += 1
            scoreboard[player_2_agent.get_name()][player_1_agent.get_name()]["loss"] += 1
            winner = player_1_agent.get_name()
            break
          if game_state == GameState.PLAYER_2_WIN:
            scoreboard[player_1_agent.get_name()][player_2_agent.get_name()]["win"] += 1
            scoreboard[player_2_agent.get_name()][player_1_agent.get_name()]["loss"] += 1
            winner = player_1_agent.get_name()
            break
          curr_player = next_player[curr_player]

        if winner != None:
          print(winner, "won!")
        else:
          print("Draw!")

      player_2_agent = agent_list[player_1_idx]
      player_1_agent = agent_list[player_2_idx]
      print("%s VS %s:" %(player_1_agent.get_name(), player_2_agent.get_name()))
      player_map = {Player.PLAYER_1 : player_1_agent, Player.PLAYER_2 : player_2_agent}

      for i in range(num_games // 2):
        print("Game %d" %(i+1))
        game = Connect4Board()
        curr_player = Player.PLAYER_1
        winner = None

        while True:
          #game.print_board()
          #print("=======================")
          game = game.add_piece(curr_player, player_map[curr_player].get_action(curr_player, game))

          game_state = game.check_game_state(curr_player)
          if game_state == GameState.DRAW:
            scoreboard[player_1_agent.get_name()][player_2_agent.get_name()]["draw"] += 1
            scoreboard[player_2_agent.get_name()][player_1_agent.get_name()]["draw"] += 1
            winner = None
            break
          if game_state == GameState.PLAYER_1_WIN:
            scoreboard[player_1_agent.get_name()][player_2_agent.get_name()]["win"] += 1
            scoreboard[player_2_agent.get_name()][player_1_agent.get_name()]["loss"] += 1
            winner = player_1_agent.get_name()
            break
          if game_state == GameState.PLAYER_2_WIN:
            scoreboard[player_1_agent.get_name()][player_2_agent.get_name()]["win"] += 1
            scoreboard[player_2_agent.get_name()][player_1_agent.get_name()]["loss"] += 1
            winner = player_1_agent.get_name()
            break
          curr_player = next_player[curr_player]

        if winner != None:
          print(winner, "won!")
        else:
          print("Draw!")

  # output counts
  print_scoreboard(scoreboard)

if __name__ == "__main__":
  num_games = 6
  if len(sys.argv) == 2:
    num_games = int(sys.argv[1])
  play_games(num_games)
