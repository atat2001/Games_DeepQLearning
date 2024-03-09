import numpy as np
import tensorflow as tf
from easier_hexagon import Game
from easier_hexagon import Player

class Step:
    def __init__(self, cur_state, move_selected, moves_instant_reward, agent_index, next_state):
        self.cur_state = cur_state
        self.move_selected = move_selected
        self.moves_instant_reward = moves_instant_reward
        self.agent_index = agent_index
        self.reward = None
        self.next_state = next_state
        

class State:
    def __init__(self):
        self.matrix1 = None
        self.matrix2 = None
        
    def _copy_state(self,state):
        # Create two empty matrices
        self.matrix1 = np.copy(state.matrix1)
        self.matrix2 = np.copy(state.matrix2)
        return self

    def _init_state(self, game, input_shape):
        # Create two empty matrices
        self.matrix1 = np.zeros(input_shape) # current game state
        self.matrix2 = np.zeros(input_shape) # available moves from index matrix2.x to index stored in matrix2.y
        for move in game.moves_idf:
            self.matrix2[move[0]][move[1]] = 1
        return self

    def apply_move(self,action):
        self.matrix1[action[0]][action[1]] = 1
        self.matrix2[action[0]][action[1]] = 0
        return

class Agent:
    # def __init__(self,input_shape1, input_shape2,num_actions):
    def __init__(self, state_shape1, state_shape2, num_actions, learning_rate=0.001, discount_factor=0.99, load_weights = True):
        self.state_shape1 = state_shape1
        self.state_shape2 = state_shape2
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.agent_index = 1
        # Define the Q-network
        self.q_network = self._build_q_network()
        self.human_input = False
        self.training_file = "final_reward_weights_backup.h5"

        # Compile the Q-network
        self.q_network.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
                               loss='mean_squared_error')
        self.will_learn = load_weights
        if(load_weights):
            self.load_weights()

    def _build_q_network(self):
        # Input layers for each matrix
        input_layer1 = tf.keras.layers.Input(shape=self.state_shape1)
        input_layer2 = tf.keras.layers.Input(shape=self.state_shape2)
        
        # Flatten the input matrices
        flattened_input1 = tf.keras.layers.Flatten()(input_layer1)
        flattened_input2 = tf.keras.layers.Flatten()(input_layer2)

        # Concatenate the flattened input matrices
        concatenated_inputs = tf.keras.layers.Concatenate()([flattened_input1, flattened_input2])

        # Define neural network architecture
        hidden_layer1 = tf.keras.layers.Dense(128, activation='relu')(concatenated_inputs)
        hidden_layer2 = tf.keras.layers.Dense(64, activation='relu')(hidden_layer1)
        output_layer = tf.keras.layers.Dense(self.num_actions)(hidden_layer2)

        # Create the model
        model = tf.keras.models.Model(inputs=[input_layer1, input_layer2], outputs=output_layer)
        return model


    def select_actions(self, state):

        # Convert the state matrices to NumPy arrays
        state_matrix1_array = np.array(state.matrix1)
        state_matrix2_array = np.array(state.matrix2)

        # Expand the dimensions to match the input shape expected by the Q-network
        state_matrix1_expanded = np.expand_dims(state_matrix1_array, axis=0)
        state_matrix2_expanded = np.expand_dims(state_matrix2_array, axis=0)
        return self.q_network.predict([state_matrix1_expanded, state_matrix2_expanded],verbose = 0)[0]

    # step list -> batch
    def train(self, batch):
        if(not(self.will_learn)):
            return
        # Extract data from the batch
        # print([step.reward for step in batch])
        # exit()
        batch_states1 = np.array([step.cur_state.matrix1 for step in batch if step.reward != None])
        batch_states2 = np.array([step.cur_state.matrix2 for step in batch if step.reward != None])
        batch_next_states1 = np.array([step.next_state.matrix1 for step in batch if step.reward != None])
        batch_next_states2 = np.array([step.next_state.matrix2 for step in batch if step.reward != None])
        batch_rewards = np.array([step.reward for step in batch if step.reward != None])
        batch_actions = np.array([step.move_selected for step in batch if step.reward != None]) #  and step.reward != None
        print(f"skipping {len([step for step in batch if step.reward == None])}")
        # Compute target Q-values
        target_q_values = batch_rewards + self.discount_factor * np.max(
            self.q_network.predict([batch_next_states1, batch_next_states2]), axis=1)

        # Compute current Q-values
        current_q_values = self.q_network.predict([batch_states1, batch_states2])

        # Update Q-values for the actions taken
        for i, action in enumerate(batch_actions):
            current_q_values[i][action] = target_q_values[i]

        # Train the Q-network on the batch
        self.q_network.fit([batch_states1, batch_states2], current_q_values, verbose=0)

    def load_weights(self):
        if(not(self.load_weights)):
            return
        try:
            string = self.training_file
            self.q_network.load_weights(string)
            print("loaded weights")
        except Exception as e:
            print("Error:", e)

    def save_weights(self):
        if(not(self.load_weights)):
            return
        try:
            string = self.training_file
            self.q_network.save_weights(string)
            print("saved weights")
        except Exception as e:
            print("Error:", e)
    
    def set_training_file(self, file):
        self.training_file = file


class Trainer:
    def __init__(self, input_shape, agents, game, player1_name = "player1", player2_name = "player2"):
        self.input_shape = input_shape       # set this now
        self.step_list = []    #   [(cur_state, move_selected, moves instant reward, agent_index,)]
        self.agents = agents
        self.game = game
        self.agent_index = 0
        self.nr_agents = 2 # fixed for now
        self.cur_state = State()._init_state(self.game, self.input_shape)
        self.actions = game.get_moves_idf() # change to general
        self.filter = []
        self.batch_size = game.get_number_of_moves()*10 + 1     # aumentar com o tempo
        self.batch_index = 0
        self.save = 0
        self.cur_game = []
        self.last_scores = []
        self.player1_name = player1_name
        self.player2_name = player2_name
        self.last_score = []
        # set the agent_index for the agents
        """
        for a_index in range(len(agents)):
            self.agents[a_index] = self.agents[0]
        """
    def reset_game(self):
        del self.game
        self.game = Game(3, Player(self.player1_name), Player(self.player2_name))
        self.filter = []
        self.cur_state = State()._init_state(self.game, self.input_shape)
        # self.step_liststep_list = [] 

    def select_move(self):
        action = self.agents[self.agent_index].select_actions(self.cur_state)
        for index in self.filter:
            action[index] = -100
        # Epsilon-greedy policy for action selection
        # EPSILON
        if np.random.rand() < 0.1 and self.agents[self.agent_index].human_input == False:

            random_list = [i for i, x in enumerate(action) if x != -100]
            # Random action
            return np.random.choice(random_list)
        action = np.argmax(action)
        return action

    def get_last_step(self):
        return self.step_list[-1]

    def save_step(self, step):
        if(self.batch_index < self.batch_size):
            self.step_list = self.step_list + [step]
            self.batch_index += 1
        else:
            self.teach_agent()
        
    def measure_accuracy(self):
        returner = 0
        x = len(self.last_scores)
        if(x > 100):
            x = 1
            self.last_scores = [self.last_scores[-1]]
            return 
        for score_pair in self.last_scores:
            returner += abs(score_pair[0] - score_pair[1])
        #print(f"{score_pair[0]} {score_pair[1]}")
        #print(f"{returner} / {len(self.last_scores)} * 24")
        return 1-(returner/(x*24))

    def step(self):
        if(self.game.check_game_finished()):
            self.init_rewards_in_step_list()
            self.cur_game = []
            self.last_score = self.game.score
            self.reset_game()
            print(self.measure_accuracy())
        # [(cur_state, move_selected, moves_instant_reward, agent_index,)]
        move_selected = self.select_move()
        state_copy = State()._copy_state(self.cur_state)
        self.filter = self.filter + [move_selected]
        moves_instant_reward = self.game.make_action(self.actions[move_selected])
        #print("actions left:")
        #print([self.actions[x] for x in range(len(self.actions)) if x not in self.filter])
        #print("\n\n")
        self.cur_state.apply_move(self.actions[move_selected])

        self.save_step(Step(state_copy, move_selected, moves_instant_reward, self.agent_index, State()._copy_state(self.cur_state)))
        self.cur_game += [self.step_list[-1]]

        if(moves_instant_reward == 0):
            self.agent_index = (self.agent_index + 1) % self.nr_agents
        if(moves_instant_reward == -1):
            print(self.actions[move_selected])
            print("error, reward -1???")
            exit()

    def play_full_game(self):
        while not(self.game.check_game_finished()):
            self.step()
        print("game done")
    
    def init_rewards_in_step_list(self):
        # [cur_state, move_selected, moves_instant_reward, agent_index]
        scores = np.zeros(self.nr_agents)
        for step in self.cur_game:
            scores[step.agent_index] += step.moves_instant_reward
        whole_score = [0,0]
        whole_score[0] = scores[0] - scores[1]  # simplified for 2, for more could be my_score-max_score
        whole_score[1] = scores[1] - scores[0]
        self.last_scores = self.last_scores + [[scores[0], scores[1]],]
        self.game.setScore([scores[0], scores[1]])
        cur_score = [0,0]
        for step in self.cur_game:
            cur_score[step.agent_index] += step.moves_instant_reward
            step.reward = whole_score[step.agent_index] - (cur_score[step.agent_index] - cur_score[step.agent_index - 1])   # the score difference in the end should be bigger than right now basicly
        
    def teach_agent(self):
        self.batch_index = 0
        self.save += 1
        self.agents[0].train(self.step_list[:-1])
        for agent in range(len(self.agents)):
            self.agents[agent] = self.agents[0]

        if(self.save == 10):
            self.save = 0
            self.agents[0].save_weights()
        self.step_list = [self.step_list[-1]]
        return 

    def learning_cycle(self):
        self.play_full_game()
        self.init_rewards_in_step_list()
        self.teach_agent()

def apply_move(action, matrix1, matrix2):
    for index in range(3):
        if matrix2[action[0]][index] == action[1]:
            matrix1[action[0]][index] = action[1]
            matrix2[action[0]][index] = 0
    return
