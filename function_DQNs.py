# import the necessary libraries
import numpy as np
import random
import tensorflow as tf
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.layers import Input, Dense
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.optimizer_v2.adam import Adam
from collections import deque
from tensorflow import gather_nd
from tensorflow.python.keras.losses import mean_squared_error
import math
import os
from Environment_FLP.plotting import plots
from tensorflow.python.keras import backend as K


class DeepQLearning:

    ###########################################################################
    #   START - __init__ function
    ###########################################################################
    # INPUTS:
    # env - FLP environment
    # gamma - discount rate
    # epsilon - parameter for epsilon-greedy approach
    # numberEpisodes - total number of simulation episodes

    def __init__(self, env, network_type, alpha, gamma, epsilon, layer_units, numberEpisodes, trial_length):

        # # Fixing seeds for reproducibility
        np.random.seed(42)
        tf.keras.utils.set_random_seed(42)  # sets seeds for base-python, numpy and tf

        self.env = env
        self.action_dict = self.env.get_action_name_dict()
        self.gamma = gamma
        self.epsilon = epsilon
        self.alpha = alpha
        self.numberEpisodes = numberEpisodes

        self.trial_length = trial_length

        # Initialize a list to store accuracies
        self.accuracies = []

        # Initialize a list to store losses
        self.losses = []

        # state dimension
        self.state_dim = self.env.observation_space.shape[0]

        # action dimension
        self.action_dim = self.env.action_space.n

        # this is the maximum size of the replay buffer
        self.replayBufferSize = 1000

        # this is the size of the training batch that is randomly sampled from the replay buffer
        self.batchReplayBufferSize = 400

        # this sum is used to store the sum of rewards obtained during each training episode
        self.Rewards_per_Episode = []

        # Track the total count for each action
        self.total_action_count = []

        # replay buffer
        self.replayBuffer = deque(maxlen=self.replayBufferSize)

        # Choose the type of DQN network to create
        self.network_type = network_type

        # number of layers and units:
        self.layer_units = layer_units

        # that is, every updateTargetNetworkPeriod we update the target network parameters
        self.updateTargetNetworkPeriod = trial_length

        # parameters and reset the counter to zero, this process is repeated until the end of the training process
        self.counterUpdateTargetNetwork = 0

        # this list is used in the cost function to select certain entries of the
        # predicted and true sample matrices in order to form the loss
        self.actionsAppend = []

        # create folder base
        self.folder_name = f'{network_type}'
        self.folder_network_execution = f"{self.network_type}_count_{self.numberEpisodes}"

        # Create the main network
        # Create the network based on the chosen type
        if self.network_type == "DQN":
            self.mainNetwork = self.Create_DQN_Network(self.layer_units)
            self.targetNetwork = None
        elif self.network_type == "DDQN":
            self.mainNetwork = self.Create_DQN_Network(self.layer_units)
            self.targetNetwork = self.Create_DQN_Network(self.layer_units)
        elif self.network_type == "Dueling DQN":
            self.mainNetwork = self.Create_Dueling_Network(self.layer_units)
            self.targetNetwork = None
        elif self.network_type == "Double Dueling DQN":
            self.mainNetwork = self.Create_Dueling_Network(self.layer_units)
            self.targetNetwork = self.Create_Dueling_Network(self.layer_units)
        else:
            raise ValueError("Invalid network type")

    ###########################################################################
    #   END - __init__ function
    ###########################################################################

    ###########################################################################
    # START - function for defining the loss (cost) function
    # INPUTS:
    #
    # y_true - matrix of dimension (self.batchReplayBufferSize,2) - this is the target
    # y_pred - matrix of dimension (self.batchReplayBufferSize,2) - this is predicted by the network
    #
    # - this function will select certain row entries from y_true and y_pred to form the output
    # the selection is performed on the basis of the action indices in the list  self.actionsAppend
    # - this function is used in createNetwork(self) to create the network
    #
    # OUTPUT:
    #
    # - loss - watch out here, this is a vector of (self.batchReplayBufferSize,1),
    # with each entry being the squared error between the entries of y_true and y_pred
    # later on, the tensor flow will compute the scalar out of this vector (mean squared error)

    def my_loss_fn(self, y_true, y_pred):
        s1, s2 = self.batchReplayBufferSize, 2
        indices = np.zeros(shape=(s1, s2))
        indices[:, 0] = np.arange(s1)
        indices[:, 1] = self.actionsAppend[-s1:]
        predicted_labels = gather_nd(y_pred, indices=indices.astype(int))
        true_labels = gather_nd(y_true, indices=indices.astype(int))
        loss = mean_squared_error(predicted_labels, true_labels)
        return loss

    #   END - of function my_loss_fn
    ###########################################################################
    ###########################################################################
    #   START - function createNetwork()
    # this function creates the network
    ###########################################################################
    # Reset the parameters
    @staticmethod
    def reset_parameters(model):
        session = K.get_session()
        for layer in model.layers:
            if hasattr(layer, 'kernel') and layer.kernel is not None:
                layer.kernel.initializer.run(session=session)
            if hasattr(layer, 'bias') and layer.bias is not None:
                layer.bias.initializer.run(session=session)

    def Create_DQN_Network(self, layer_units):
        model = Sequential()
        # Input layer
        model.add(Dense(layer_units[0], input_dim=self.state_dim, activation='relu'))

        # Hidden layers based on the layer_units list
        for units in layer_units[1:]:
            model.add(Dense(units, activation='relu'))

        # Output layer
        model.add(Dense(self.action_dim, activation='linear'))

        # Compile the network with the custom loss function
        model.compile(optimizer=Adam(lr=self.alpha), loss=self.my_loss_fn, metrics=['accuracy'])

        return model

    def Create_Dueling_Network(self, layer_units):
        X_input = Input(shape=(self.state_dim,))
        X = X_input

        # Hidden layers based on the layer_units list
        for units in layer_units:
            X = Dense(units, activation="relu")(X)

        # State value branch
        state_value = Dense(1)(X)

        # Action advantage branch
        action_advantage = Dense(self.action_dim)(X)

        # Combining state value and action advantage to compute Q-values
        X = state_value + (action_advantage - tf.math.reduce_mean(action_advantage, axis=1, keepdims=True))

        # Create the model
        model = Model(inputs=X_input, outputs=X)
        model.compile(loss=self.my_loss_fn, optimizer=Adam(lr=self.alpha), metrics=['accuracy'])

        return model

    ###########################################################################
    #   END - function createNetwork()
    ###########################################################################

    ###########################################################################
    #   START - function trainingEpisodes()
    #   - this function simulates the episodes and calls the training function
    #   - trainNetwork()
    ###########################################################################

    def trainingEpisodes(self):
        # reset the environment at the beginning of every episode

        # here we loop through the episodes
        for indexEpisode in range(self.numberEpisodes):
            # list that stores rewards per episode - this is necessary for keeping track of convergence
            rewardsEpisode = []

            print("Simulating episode {}".format(indexEpisode))
            currentState, past_cost = self.env.reset()
            # here we step from one state to another
            # this will loop until a terminal state is reached
            terminalState = False
            while not terminalState:
                # select an action on the basis of the current state, denoted by currentState
                action = self.selectAction(self.action_dict, currentState, indexEpisode)

                # here we step and return the state, reward, and boolean denoting if the state is a terminal state
                (nextState, reward, terminalState) = self.env.step(action)
                rewardsEpisode.append(reward)

                # add current state, action, reward, next state, and terminal flag to the replay buffer
                self.replayBuffer.append((currentState, action, reward, nextState, terminalState))

                # train network
                self.trainNetwork()

                # set the current state for the next step
                currentState = nextState
                # set the current cost to the new cost

                if terminalState:
                    print(f'Final reward in the episodes {indexEpisode}:', reward)

            # print the sum of rewards
            print("Sum of rewards {}".format(np.sum(rewardsEpisode)))
            self.Rewards_per_Episode.append(np.sum(rewardsEpisode))

        ############################################################################

        # Create the folder
        # Specify the base folder name
        try:
            os.mkdir(self.folder_name)
            print(f"Folder '{self.folder_name}' created successfully.")
        except OSError as e:
            print(f"Error creating the folder: {e}")
        # Create the directory if it doesn't exist
        # Create the directory for folder_network if it doesn't exist
        folder_network_path = os.path.join(self.folder_name, self.folder_network_execution)
        os.makedirs(folder_network_path, exist_ok=True)

        # plotting of the accuracies, losses and rewards
        folder = os.path.join(self.folder_name, self.folder_network_execution)
        plots(folder, self.accuracies, 'accuracies', self.numberEpisodes, self.alpha)
        plots(folder, self.losses, 'losses', self.numberEpisodes, self.alpha)
        plots(folder, self.Rewards_per_Episode, 'Cumulative_Rewards', self.numberEpisodes, self.alpha)

    ###########################################################################
    #   END - function trainingEpisodes()
    ###########################################################################

    # ###########################################################################
    # #    START - function for selecting an action: epsilon-greedy approach
    # ###########################################################################
    # we start by implementing the decay function that it will decay to 0 in the last 100 per cent of the episodes
    # Function to calculate epsilon for a given episode

    def calculate_epsilon(self, initial_epsilon, min_epsilon, decay_rate, episode):
        self.epsilon = min_epsilon + (initial_epsilon - min_epsilon) * math.exp(-decay_rate * episode)
        return self.epsilon

    # this function selects an action on the basis of the current state
    # INPUTS:
    # state - state for which to compute the action
    # index - index of the current episode
    def selectAction(self, action_dict, state, index):
        decay_rate = self.calculate_epsilon(1, 0., 4 * 1 / self.numberEpisodes, index)
        random_number = np.random.random()
        if random_number < decay_rate:
            # Randomly select a machine and action index
            action_index = np.random.choice(list(action_dict.keys()))
            # print(machine, action_index)

        else:
            state_input = np.array([state])
            # print('machine_input', machine_input)
            Qvalues = self.mainNetwork.predict(state_input)[0]

            Qvalues = np.array(Qvalues)
            # print('Qvalues', Qvalues)
            # input('press enter to continue:',)

            # Extract the index of the action with the maximum Q-value
            action_index = np.argmax(Qvalues)

            # print('action_id', action_id)
        return action_index

    # ###########################################################################
    # #    END - function selecting an action: epsilon-greedy approach
    # ###########################################################################

    ###########################################################################
    #    START - function trainNetwork() - this function trains the network
    ###########################################################################

    def trainNetwork(self):
        if len(self.replayBuffer) >= self.batchReplayBufferSize:
            randomSampleBatch = random.sample(self.replayBuffer, self.batchReplayBufferSize)

            currentStateBatch = np.zeros(shape=(self.batchReplayBufferSize, self.state_dim))
            nextStateBatch = np.zeros(shape=(self.batchReplayBufferSize, self.state_dim))
            reward_sample = np.zeros(shape=(self.batchReplayBufferSize, 1))
            outputNetwork = np.zeros(shape=(self.batchReplayBufferSize, self.action_dim))

            for index, tupleS in enumerate(randomSampleBatch):
                currentStateBatch[index, :] = tupleS[0]
                nextStateBatch[index, :] = tupleS[3]
                reward_sample[index, :] = tupleS[2]

            # Use the main network to predict Q-values for the next state (Double DQN) if needed
            if self.network_type in ["DQN", "Dueling DQN"]:
                QnextStateMainNetwork = self.mainNetwork.predict([nextStateBatch])
                QnextStateTargetNetwork = None

            # here, use the target network to predict next-Q-values if it exists
            elif self.network_type in ["DDQN", "Double Dueling DQN"]:
                QnextStateMainNetwork = self.mainNetwork.predict([nextStateBatch])
                QnextStateTargetNetwork = self.targetNetwork.predict([nextStateBatch])

            else:
                raise ValueError("Invalid network type")

            # here, use the target network to predict Q-values
            QcurrentStateMainNetwork = self.mainNetwork.predict([currentStateBatch])

            # Use the main network to predict Q-values for the next state (Double DQN).
            inputNetwork = currentStateBatch

            for index, (currentState, action, reward, nextState, terminated) in enumerate(randomSampleBatch):
                if terminated:
                    y = reward_sample[index]
                # else:
                # Use the main network to select the action with the highest Q-value for the next state
                if self.network_type in ["DDQN", "Double Dueling DQN"]:
                    next_action = np.argmax(QnextStateMainNetwork[index])
                    y = reward_sample[index] + self.gamma * QnextStateTargetNetwork[index][next_action]
                else:
                    y = reward_sample[index] + self.gamma * np.max(QnextStateMainNetwork[index])

                self.actionsAppend.append(action)
                outputNetwork[index] = QcurrentStateMainNetwork[index]
                outputNetwork[index, action] = y
            # print(len(outputNetwork), len(self.actionsAppend))
            history = self.mainNetwork.fit(inputNetwork, outputNetwork, batch_size=self.batchReplayBufferSize,
                                           verbose=0,
                                           epochs=100)
            self.accuracies.append(history.history['accuracy'][-1])
            self.losses.append(history.history['loss'][-1])

            self.counterUpdateTargetNetwork += 1

            if self.counterUpdateTargetNetwork >= self.updateTargetNetworkPeriod:
                if self.targetNetwork is not None:
                    self.targetNetwork.set_weights(self.mainNetwork.get_weights())
                    print("Target network updated!")
                else:
                    print("No target network to update.")
                # print("Counter value {}".format(self.counterUpdateTargetNetwork))
                self.counterUpdateTargetNetwork = 0

    def save_model(self):
        # Save the model weights and architecture
        model_filename = os.path.join(self.folder_name, self.folder_network_execution)
        model_filename_modelname = os.path.join(model_filename, f'model with alpha = {self.alpha}.h5')
        self.mainNetwork.save(model_filename_modelname)

