# Games_DeepQLearning
Deep q learning for a game with the objective of making triangles.

Algorithm frontend lets you play vs the ml agent. The one provided is not trained with much data.

The algorithm_trainer lets you train the Agent (playing against itself to collect data) that is defined in the algorithm2 file.

To train the algorithm we use a reward function that takes in the games overall score - the current score, this is to ensure the algorithm chooses the option that maximizes
the points gained in a certain position keeping in mind the long term affects of such a move. To make sure it is learning I used a function that evaluates how close each game
is (from 0 to 1, 1 being a tied game, 0 a game where somone didnt score a point), and it does an average over the last x games, then it prints that average value to see if the
games are getting closer. 
I havent trained the model enought to know if it is getting better.  

I belive for this reward function to work (since it thinks in the long term) it would require a way bigger training set, so dont expect the algorithm to play well...


Files of interest:

easier_hexagon  -> has the game implementation

algorithm_frontend.py -> lets you play the game vs another player or vs the Ai if we change a few lines

algorithm2 -> has trainer and the ai model

algo_vs_old_algo -> algorithm plays with a untrained version

algorithm_trainer -> trains the Ai using algorithm2



