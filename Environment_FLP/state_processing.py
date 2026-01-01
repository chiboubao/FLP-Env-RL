import numpy as np
import tensorflow as tf
from tensorflow.python.keras.layers import Dense, Input, concatenate
from tensorflow.python.keras.models import Model


# Set fixed seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)


def processing_states():
    # Define input layers for coordinates_input and transport_input
    input_coordinates = Input(shape=(16,))
    input_transport = Input(shape=(16,))

    # Process coordinates_input
    X_coord = Dense(16, activation='sigmoid')(input_coordinates)
    X_coord = Dense(8)(X_coord)

    # Process transport_input
    X_transport = Dense(16, activation='sigmoid')(input_transport)
    X_transport = Dense(8)(X_transport)

    # Concatenate processed streams
    combined_state = concatenate([X_coord, X_transport])

    model = Model(inputs=[input_coordinates, input_transport], outputs=combined_state)
    return model


def get_state(pos, coord):
    model = processing_states()
    # Convert inputs to numpy arrays
    pos_array = np.array(pos)
    demand_array = np.array(coord)
    # Reshape the arrays to match the input shapes
    pos_array = pos_array.reshape(1, -1)  # Reshape to (1, 16)
    demand_array = demand_array.reshape(1, -1)  # Reshape to (1, 16)
    # make output
    output = model.predict([pos_array, demand_array])
    return output


def normalize_position_nodes(vector, min_val: int, max_val: int):
    """
    Normalize a vector between 0 and 1.

    Args:
    vector (list): The input vector.

    Returns:
    list: The normalized vector.
    """
    normalized_vector = [(x - min_val) / (max_val - min_val) for x in vector]
    return normalized_vector


def normalize_state(state, seed=4):
    np.random.seed(seed)  # Set the seed for reproducibility
    # print('state:', state)
    state = np.array(state)  # Ensure the state is a numpy array
    scaler = MinMaxScaler()
    state_normalized = scaler.fit_transform(state.reshape(-1, 1)).flatten()
    return state_normalized


def get_state_without_machines(positions):
    positions_normalized = normalize_position_nodes(positions, 0, 6)
    final_state_array = np.array(positions_normalized)
    return final_state_array


def get_state(positions, df):
    positions_normalized = normalize_position_nodes(positions, 0, 6)
    # Sample DataFrame

    # Convert the DataFrame to a one-dimensional vector
    vector = df.values.flatten()

    # Convert to a list if needed
    vector_list = vector.tolist()

    # Concatenate the two lists into one
    state_list = positions_normalized + vector_list

    state = np.array(state_list)

    return state


def get_state_normalized(positions, df):
    positions_normalized = normalize_position_nodes(positions, 0, 6)

    # Convert the DataFrame to a one-dimensional vector
    demand = df.values.flatten()

    # Convert to a list if needed
    demand_list = demand.tolist()
    demand_list_normalized = normalize_state(demand_list).tolist()
    # Concatenate the two lists into one
    state = positions_normalized
    # print('State:', state)
    return state


# # Example usage:
# pos = [1, 2, 3, 4, 5, 6, 1, 2, 3, 5, 5, 6, 2, 0, 2, 0]
# demand = [0, 0, 0, 40, 0, 10, 10, 0, 0, 0, 0, 40, 10, 0, 0, 0]
# state = get_state(pos, demand)
# print(state)
