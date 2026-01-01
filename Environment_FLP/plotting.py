import itertools
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os
plt.rcParams['font.family'] = 'DejaVu Serif'


def plotting_interactive(G, reference_node, cost):
    plt.figure(1)  # Adjust width and height as needed
    pos = nx.get_node_attributes(G, 'pos')
    color_map = []
    colored_node = [m for m in reference_node if m in G.nodes]
    for node in G.nodes:
        if node in colored_node:
            color_map.append('blue')
        else:
            color_map.append('red')

    green_nodes = [node for node in G.nodes if not node.startswith(('P', 'D')) and int(node) not in reference_node]
    red_nodes = [node for node in G.nodes if node.startswith(('P', 'D'))]
    blue_nodes = [str(node) for node in reference_node]
    # Draw the graph with nodes and edges in default colors
    nx.draw(G, pos=pos, with_labels=True,
            node_color='white', node_size=1500, font_color='white', font_size=20, font_family='DejaVu Serif',
            edge_color='grey', width=5)

    # Draw green nodes on top to highlight them
    nx.draw_networkx_nodes(G, pos, nodelist=green_nodes, node_color='green', node_size=1500)
    nx.draw_networkx_nodes(G, pos, nodelist=blue_nodes, node_color='blue', node_size=1500)
    nx.draw_networkx_nodes(G, pos, nodelist=red_nodes, node_color='red', node_size=1500)
    # print('green_nodes', green_nodes, reference_node)
    # Draw edge labels
    edge_label = {(u, v): (d['weight']) for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos=pos, edge_labels=edge_label, font_color='black', font_size=20,
                                 font_family='DejaVu Serif')
    # Add the cost variable to the plot
    plt.text(0.2, 0.8, f'Cost: {cost}', horizontalalignment='left', verticalalignment='top', transform=plt.gca().transAxes, fontsize=15, fontfamily='DejaVu Serif', bbox=dict(facecolor='white', alpha=0.8))
    plt.draw()
    plt.pause(0.5)


def modify_keys_with_same_values(dictionary):
    # Initialize an empty dictionary to store keys with same values
    same_values = {}

    # Iterate through the dictionary
    for key, value in dictionary.items():
        # Convert the value to a tuple to make it hashable
        value_tuple = tuple(value)
        # Check if the value exists as a key in the same_values dictionary
        if value_tuple in same_values:
            # Append the current key to the list of keys with the same value
            same_values[value_tuple].append(key)
        else:
            # Create a new list with the current key as the first element
            same_values[value_tuple] = [key]

    # Filter out values that have only one key
    same_values = {k: v for k, v in same_values.items() if len(v) > 1}

    # If there are keys with the same values, modify the keys in the original dictionary
    if same_values:
        for value, keys in same_values.items():
            # Join the keys with '/'
            new_key = '/'.join(keys)
            # Modify all keys in the original dictionary
            for key in keys:
                # Don't change the values, only the keys
                dictionary[new_key] = dictionary.pop(key)


def plotting_interactive_final(G, reference_node):
    plt.figure(1)
    pos = nx.get_node_attributes(G, 'pos')
    positions = pos
    modify_keys_with_same_values(positions)
    color_map = []
    colored_node = [m for m in reference_node if m in G.nodes]
    for node in G.nodes:
        if node in colored_node:
            color_map.append('blue')
        else:
            color_map.append('red')

    green_nodes = [node for node in G.nodes if not node.startswith(('P', 'D')) and int(node) not in reference_node]
    red_nodes = [node for node in G.nodes if node.startswith(('P', 'D'))]
    blue_nodes = [str(node) for node in reference_node]
    print(positions)
    # Draw the graph with nodes and edges in default colors
    nx.draw(G, pos=positions, with_labels=True,
            node_color='white', node_size=1500, font_color='white', font_size=20, font_family='Times New Roman',
            edge_color='grey', width=5)

    # Draw green nodes on top to highlight them
    nx.draw_networkx_nodes(G, positions, nodelist=green_nodes, node_color='green', node_size=1500)
    nx.draw_networkx_nodes(G, positions, nodelist=blue_nodes, node_color='blue', node_size=1500)
    nx.draw_networkx_nodes(G, positions, nodelist=red_nodes, node_color='red', node_size=2000)
    # print('green_nodes', green_nodes, reference_node)
    # Draw edge labels
    edge_label = {(u, v): (d['weight']) for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos=positions, edge_labels=edge_label, font_color='black', font_size=20,
                                 font_family='Times New Roman')
    plt.draw()
    plt.pause(1)


def plotting_interactive_saving(G, reference_node, episode, step, cost):
    plt.figure(1)
    pos = nx.get_node_attributes(G, 'pos')
    color_map = []
    colored_node = [m for m in reference_node if m in G.nodes]
    for node in G.nodes:
        if node in colored_node:
            color_map.append('blue')
        else:
            color_map.append('red')

    green_nodes = [node for node in G.nodes if not node.startswith(('P', 'D')) and int(node) not in reference_node]
    red_nodes = [node for node in G.nodes if node.startswith(('P', 'D'))]
    blue_nodes = [str(node) for node in reference_node]
    # Draw the graph with nodes and edges in default colors
    nx.draw(G, pos=pos, with_labels=True,
            node_color='white', node_size=1500, font_color='white', font_size=20, font_family='Times New Roman',
            edge_color='grey', width=5)

    # Draw green nodes on top to highlight them
    nx.draw_networkx_nodes(G, pos, nodelist=green_nodes, node_color='green', node_size=1500)
    nx.draw_networkx_nodes(G, pos, nodelist=blue_nodes, node_color='blue', node_size=1500)
    nx.draw_networkx_nodes(G, pos, nodelist=red_nodes, node_color='red', node_size=1500)
    # print('green_nodes', green_nodes, reference_node)
    # Draw edge labels
    edge_label = {(u, v): (d['weight']) for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos=pos, edge_labels=edge_label, font_color='black', font_size=20,
                                 font_family='Times New Roman')
    plt.text(0.23, 0.73, f'Cost: {cost}', horizontalalignment='left', verticalalignment='top', transform=plt.gca().transAxes, fontsize=15, fontfamily='Times New Roman', bbox=dict(facecolor='white', alpha=0.8))
    plt.savefig("figure" + str(step) + ".png")
    # Add the cost variable to the plot
    plt.draw()
    plt.pause(10)


def plots(folder_name, data_to_plot, data_name, numberEpisodes, alpha):
    # Plot and save accuracy
    # Extract data based on data type
    fig = plt.figure(1)
    plt.plot(data_to_plot)
    plt.xlabel('steps')
    plt.ylabel(str(data_name))
    plt.title(f'{data_name}_for_{numberEpisodes}_episodes_and_learning_rate{alpha}.png')
    accuracy_plot_path = os.path.join(folder_name,
                                      f'{data_name}_for_{numberEpisodes}_episodes_and_learning_rate_{alpha}.png')
    plt.savefig(accuracy_plot_path, dpi=fig.dpi)
    plt.close()


def comparison_function(episodes, network1, network2, network3, network4, save=None):
    # Load data from the text file
    data_1 = np.loadtxt(f"comparison_plots_{episodes}/Rewards_per_Episode_{network1}_{episodes}.txt")
    data_2 = np.loadtxt(f"comparison_plots_{episodes}/Rewards_per_Episode_{network2}_{episodes}.txt")
    data_3 = np.loadtxt(f"comparison_plots_{episodes}/Rewards_per_Episode_{network3}_{episodes}.txt")
    data_4 = np.loadtxt(f"comparison_plots_{episodes}/Rewards_per_Episode_{network4}_{episodes}.txt")

    # Assuming data columns represent DQN, DDQN, Dueling DQN, and Double Dueling DQN respectively
    episodes_range = np.arange(1, len(data_3) + 1)  # Assuming each row represents an episode

    # Plotting
    plt.figure(figsize=(10, 6))  # Adjust the figure size as needed
    plt.plot(episodes_range, data_1, label=network1)
    plt.plot(episodes_range, data_2, label=network2)
    plt.plot(episodes_range, data_3, label=network3)
    plt.plot(episodes_range, data_4, label=network4)

    # Add labels and title
    plt.xlabel('Episodes')
    plt.ylabel('Rewards')
    plt.title('Cumulative Rewards per Episode for Deep Q-learning variants')
    plt.legend()

    # Show plot
    plt.grid(False)
    # plt.show()
    if save:
        plt.savefig(f'C:/Users/achiboub001/Downloads/pythonProject/DeepQN_comparative_study/comparison_plots_{episodes}/rewards_for_{episodes}.png', dpi=500)


def plot_reward(episodes, network1, network2, network3, network4, save=None):
    # Load data from the text file
    data_1 = np.loadtxt(f"1st state presentation/comparison_plots_{episodes}/Rewards_per_Episode_{network1}_{episodes}.txt")
    data_2 = np.loadtxt(f"1st state presentation/comparison_plots_{episodes}/Rewards_per_Episode_{network2}_{episodes}.txt")
    data_3 = np.loadtxt(f"1st state presentation/comparison_plots_{episodes}/Rewards_per_Episode_{network3}_{episodes}.txt")
    data_4 = np.loadtxt(f"1st state presentation/comparison_plots_{episodes}/Rewards_per_Episode_{network4}_{episodes}.txt")

    # Assuming data columns represent DQN, DDQN, Dueling DQN, and Double Dueling DQN respectively
    episodes_range = np.arange(1, len(data_3) + 1)  # Assuming each row represents an episode

    for data, net in zip([data_4, data_3, data_2, data_1], [network4, network3, network2, network1]):
        # Plotting
        plt.figure(figsize=(10, 6))  # Adjust the figure size as needed
        plt.plot(episodes_range, data, label=net)

        # Add labels and title
        plt.xlabel('Episodes')
        plt.ylabel('Rewards')
        plt.title(f'Cumulative Rewards per Episode for {net}')
        plt.legend()

        # Show plot
        plt.grid(False)
        # plt.show()
        if save:
            plt.savefig(f'C:/Users/achiboub001/Downloads/pythonProject/DeepQN_comparative_study/1st state presentation/{net}_0.0001_40_128_128_count_{episodes}/sumRewardsEpisode_for_{episodes}_episodes_and_learning_rate_0.0001.png', dpi=500)
    #     input_file = f'C:/Users/achiboub001/Downloads/pythonProject/DeepQN_comparative_study/{network}_0.0001_40_128_128_count_{episode}'
