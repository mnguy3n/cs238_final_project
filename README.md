# CS238 Final Project - Playing Connect 4 With RL
## Infra Code
Connect 4 Environment: connect4.py
Tool to play a single game between two agents: play_single_game.py
Tool to play multiple games between multiple agents and record scores: play_multiple_games.py
## Approaches Used
### Minimax (Baseline)
Game-Playing Agent: minimax_agent.py
Tool to find evaluation function params: search_minimax_params.py

### Online Q-Learning
Game-Playing Agent: q_learning_agent.py
Converter from Connect 4 board and player to basis vector: basis_function.py
Q-Learning to find values of theta: online_approx_q_learning.py
Theta values found: theta_values.txt

### Offline Q-Learning
Game-Playing Agent: q_learning_agent.py
Converter from Connect 4 board and player to basis vector: basis_function.py
Q-Learning to find values of theta: offline_approx_q_learning.py
Episode generator that dumps data to csv: generate_episode_data.py
Data DirectorY: game_data/
Theta values found: theta_values.txt

### Forward Search
Agent: forward_search_agent.py

### Monte Carlo Tree Search
Agent: monte_carlo_tree_search_agent.py
Agent with Q-Learning: monte_carlo_tree_search_agent_with_q_value.py

## Potential Future Approaches
Deep Q-Learning
AlphaGo/AlphaGo Zero
