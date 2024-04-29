# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 15:06:03 2023

@author: johne
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns
import random
import copy
import pandas as pd
#import geopandas
#import folium


# graph representation of an environment that an agent interacts in
class Environment:
    def __init__(self, x, y):
        # the environment boundries
        self.limit_x = x
        self.limit_y = y
        # create a grid
        self.create_grid()
        # create an adacency matrix that includes diagonals 
        self.create_adjacency_matrix()
        self.adjacency_matrix = self.create_adjacency_matrix()
        # how does node effect sensor range
        # self.sensor_range_node
        # how does node effect Pr to detect
        # self.sensor_detect_node
        #how is movement effected by node
        # self.move_rate_node
    
    # create a X by Y grid with numbered nodes from 0..(X*Y)-1
    def create_grid(self):
       
        x = np.arange(0, self.limit_x, 1)
        y = np.arange(0, self.limit_y, 1)
        
        gx, gy = np.meshgrid(x, y)
        
        self.node_matrix = gx + (gy*self.limit_y)
      
    # Create a NumPy array adjacency matrix for a grid of X by Y
    def create_adjacency_matrix(self):
        rows, cols = self.node_matrix.shape
        nodes = self.node_matrix.flatten()
        adjacency_matrix = np.zeros((rows * cols, rows * cols), dtype=int)

        for b in range(len(nodes)):
            for a in range(len(nodes)):
                x1, y1 = self.XY_coord(nodes[b])
                x2, y2 = self.XY_coord(nodes[a])
                if abs(x1 - x2) <= 1 and abs(y1 - y2) <= 1:
                    adjacency_matrix[b][a] = 1

        return adjacency_matrix
    
    # Convert a node number to X, Y coordinates
    def XY_coord(self, node):
        x = node // self.limit_y
        y = node % self.limit_y
        
        return x, y

# agent class with an id, move rate, sensor range, in a node in an environment
# an agent knows the environment it is in, what an agent about he environment
# is mutable 
class Agent:
    def __init__ (self, agent_id, move_rate, sensor_range, sensor_detect_Pr, the_node, the_environment):
        self.agent_id = agent_id
        self.move_rate = move_rate
        self.sensor_range = sensor_range  
        self.sensor_detect_Pr = sensor_detect_Pr
        # where agent is in environment
        self.node = the_node
        # the environment the agent is in
        self.the_environment = the_environment
        # the path in the environment the agent has taken
        self.path = self.initialize_path()
        self.initialize_path()
        # the matrix recording the number of times the agent has been in a certain area
        self.create_search_matrix()
        self.search_matrix = self.create_search_matrix()
 
    def initialize_path(self):
        self.path = []
        self.path.append(self.node)
 
    def create_search_matrix(self):
        search_matrix = np.zeros((self.the_environment.limit_x, self.the_environment.limit_y))
        x, y = self.XY_coord(self.node)
        search_matrix[x, y] += 1
        
        return search_matrix
     
    # Convert a node number to X, Y coordinates
    def XY_coord(self, node):
        x = node // self.the_environment.limit_y
        y = node % self.the_environment.limit_y
        
        return x, y

    def move_to(self, node):
        self.path.append(node)
        x, y = self.XY_coord(node)
        self.search_matrix[x, y] += 1
        self.node = node
 
# Convert a node number to X, Y coordinates
def XY_coord(n, lim_x, lim_y):
    x = n // lim_y
    y = n % lim_y
            
    return x, y

# Convert X, Y coordinates to a node in a given environment
def node_from_XY(x, y, e):
    
    return e.node_matrix[x, y]

# given a path update a search matrix
def path_to_search_matrix(the_agent):
    # initialize a search matrix
    the_search_matrix = np.zeros((the_agent.the_environment.limit_x, the_agent.the_environment.limit_y))
    
    # update search matrix with path
    for node in the_agent.path:
        x, y = XY_coord(node, the_agent.the_environment.limit_x, the_agent.the_environment.limit_y)
        the_search_matrix[x, y] += 1
    
    return the_search_matrix
    
# create graph G using networkx library from environment
def create_graph(the_environment):
    G = nx.from_numpy_array(the_environment.adjacency_matrix)
    
    return G

# show a networkx graph
def show_graph(a_graph):
    
    print(a_graph)

    # Calculate the layout 
    pos = nx.nx_agraph.graphviz_layout(a_graph)

    # Now, draw the graph with labels
    nx.draw_networkx(a_graph, pos=pos, with_labels=True, node_size=150, alpha=.75)

    # Display the plot
    plt.show()

# score path = the number of nodes to search (n x m) minus the moves it takes
# plus the efficiency in the search (how many nodes did the agent search only once)
# perfect score is n x m
def score_path(the_agent):
    
    # efficiency = number of times searched area only once
    eff = Num_in_Matrix(1, the_agent.search_matrix)
    
    x, y = the_agent.search_matrix.shape
    
    return (((x * y) - len(the_agent.path)) + eff)

# move the agent in the environment 
def move_agent_random(the_node, the_environment):

    possible_moves = []    
    nodes = the_environment.node_matrix.flatten()
    
    for i in range(len(nodes)):
        
        if (the_environment.adjacency_matrix[the_node][i] > 0):
            possible_moves.append(nodes[i])     

    # randomly select node to move to and return it
    return random.choice(possible_moves)

# move the agent in the environment to least searched area, 
# if multiple least searched areas randomly select
def move_agent_least_searched(the_node, the_agent, the_environment):

    possible_moves = []    
    nodes = the_environment.node_matrix.flatten()
    best_moves = []
    times_searched = []
    
    # get all posible moves
    for i in range(len(nodes)):
        
        if (the_environment.adjacency_matrix[the_node][i] > 0):
            possible_moves.append(nodes[i])     
    
    # score possible moves
    for j in range(len(possible_moves)):
        
        x, y = XY_coord(possible_moves[j], the_environment.limit_x, the_environment.limit_y) 
        times_searched.append(the_agent.search_matrix[x, y])
    
    least_searched = min(times_searched)
    
    for k in range(len(possible_moves)):
        if (times_searched[k] == least_searched):
            best_moves.append(possible_moves[k])
        
    # randomly select node to move to and return it
    return random.choice(best_moves)

# move the agent in the environment to area where information gain highest, 
# if multiple least searched areas randomly select
def move_agent_info_gain(the_node, the_agent, the_environment):

    possible_moves = []    
    nodes = the_environment.node_matrix.flatten()
    best_moves = []
    times_searched = []
    
    # get all posible moves
    for i in range(len(nodes)):
        
        if (the_environment.adjacency_matrix[the_node][i] > 0):
            possible_moves.append(nodes[i])     
    
    # score possible moves
    for j in range(len(possible_moves)):
        
        x, y = XY_coord(possible_moves[j], the_environment.limit_x, the_environment.limit_y) 
        times_searched.append(the_agent.search_matrix[x, y])
    
    least_searched = min(times_searched)
    
    for k in range(len(possible_moves)):
        if (times_searched[k] == least_searched):
            best_moves.append(possible_moves[k])
        
    # randomly select node to move to and return it
    return random.choice(best_moves)

# update node visited array
def update_node_visited_array(node_list, the_agent):
    
    update_list = node_list.tolist()
    
    for node_visited in the_agent.path:
        if node_visited in update_list:
            update_list.remove(node_visited) 
    
    node_list = np.array(update_list)        
    
    return node_list
     
# sum marix
def Sum_Matrix(matrix):
    
    # Initialize a variable to store the sum
    total_sum = 0

    # Iterate through the rows and columns of the matrix
    for row in matrix:
        for element in row:
            total_sum += element
    
    return total_sum

# count how many times a value happens in a matrix
def Num_in_Matrix(value, matrix):
    
    count = 0
    
    row, col = matrix.shape
    
    for i in range(row):
        for j in range(col):
            
            if matrix[i, j] == value:
                count += 1
    
    return count

def find_path(the_agent, the_environment, to_visit, option):
    
    if option == "random":
        # print("Random option selected")
  
        while (len(to_visit) > 0):
            
            # move agent randomly
            to_node = move_agent_random(the_agent.path[-1], the_environment)
        
            # update agent search matrix and path
            the_agent.move_to(to_node)

            # update nodes_to_visit for move
            to_visit = update_node_visited_array(to_visit, the_agent)

    
    elif option == "info_gain":
        print("Info Gain option selected")
    
    elif option == "least_searched":
        #print("Least Searched option selected")
    
        while (len(to_visit) > 0):
            
            # move agent with least searched algorithm
            to_node = move_agent_least_searched(the_agent.path[-1], the_agent, the_environment)
        
            # update agent search matrix and path
            the_agent.move_to(to_node)

            # update nodes_to_visit for move
            to_visit = update_node_visited_array(to_visit, the_agent)

    else:
        print("Invalid option")
    
    return the_agent

def mutation(the_agent):
    
    # determine where to mutate path randomly 
    # must keep at least generation point and mutate at least 1 move
    path_length = len(the_agent.path)
    rp = random.randint(1, path_length-1)
    
    # update agent
    the_agent.path = the_agent.path[:-(path_length - rp)]
    the_agent.node = the_agent.path[len(the_agent.path)-1]
    the_agent.search_matrix = path_to_search_matrix(the_agent)
    
    return the_agent

def summary_results(an_agent):
    
    # print number of moves and score
    # print number of moves and score
    print("agent (generation number/agent number i.e. 13 from generation 1 agent 3)", an_agent.agent_id)
    print("number of moves: ", len(an_agent.path))
    print("search score: ", score_path(an_agent))
    print('******')
    
    # prep data for plot list that will translate node path to x, y coord path
     
    pathX = []
    pathY = []

    x_limit = an_agent.the_environment.limit_x
    y_limit = an_agent.the_environment.limit_y

    for i in range(len(an_agent.path)):
        x, y = XY_coord(an_agent.path[i], x_limit, y_limit)
        pathX.append(x)
        pathY.append(y)

    # Create a figure with a 2x1 grid of subplots
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # Create a heatmap of the matrix with the origin at (0, 0)

    # Set tick marks for the x and y axes
    x_ticks = [x for x in range(-1, x_limit)]  
    y_ticks = [x for x in range(-1, y_limit)]  

    # Plot the heatmap in the first subplot (0,0)
    sns.heatmap(an_agent.search_matrix, annot=True, cmap='coolwarm', xticklabels=False, yticklabels=False, ax=axs[0])
    axs[0].set_xlim(0, x_limit)
    axs[0].set_ylim(0, y_limit)
    axs[0].set_title('Blue Search Heatmap')

    # Plot the gridlines with a major grid every 1 unit and minor gridlines every 0.5 units
    axs[1].set_xticks([x + 0.5 for x in x_ticks], minor=True)
    axs[1].set_yticks([y + 0.5 for y in y_ticks], minor=True)
    axs[1].grid(which='minor', linestyle='--', linewidth=1, alpha=0.7)

    # Set custom x-axis and y-axis tick labels
    axs[1].set_xticks(x_ticks)
    axs[1].set_yticks(y_ticks)

    # Set axis ranges
    axs[1].set_xlim(-1, x_limit)
    axs[1].set_ylim(-1, y_limit)

    # Show paths

    # Plot the lines
    axs[1].plot(pathY, pathX, color = 'b')

    # Adjust spacing between subplots
    plt.tight_layout()

    # Display the plots/heatmap
    plt.show()

###############################################################################

# initialize environment bounds

x_limit = 5
y_limit = 5

# define grid as an environment bounded by x_limit and y_limit
grid = Environment(x_limit, y_limit)

# Reverse the rows to show it lined up with x, y coords 
reversed_grid = grid.node_matrix[::-1]

# Print the reversed matrix
print(reversed_grid)
    
print(grid.adjacency_matrix)
    
# create array of nodes to search from the defined environment
nodes_to_visit = grid.node_matrix.flatten()

print(nodes_to_visit)

# create a graph that represents the environment from networkx library and show it
G = create_graph(grid)

# use grid adjacency matrix to define a graph
G = create_graph(grid)
    
#show_graph(G)

##############################################################################
    
# initialize 

epochs = 10
num_agents = 100

# hyper parameters
algorithm_options = ("random", "info_gain", "least_searched", "random-least_searched")

# used in random least searched or random info gain mixed algorithm, 
# epsilon is % algorithm will choose to be random versus follow hueristic
epsilon = .01

# evolution keep. % of best paths kept from one generation to the next
survive = .20
survive_number = int(survive*num_agents)

# evolution mutate. % of paths after keep paths to randomly mutate from one generation to the next
mutate = .10
mutate_number = int(mutate*num_agents)

blue_agents = []
scores = []
score_list = []

for i in range(epochs):
    
    for j in range(num_agents):
        
        # reset nodes to search
        nodes_to_search = nodes_to_visit.copy()
        
        # first epoch or new random agents after survive and mutate cohorts
        if (i == 0) or (j >= (survive_number+mutate_number)):
            # define agents
            agent_id = (j*i) + j
            move_rate = 1 
            sensor_range = 1 
            sensor_detect_Pr = 1
            #start in random location in environment
            start_x = random.randint(0, x_limit-1)
            start_y = random.randint(0, y_limit-1)

            # define blue agent in the grid environment starting at start_x, start_y
            blue_agents.append(Agent(agent_id, move_rate, sensor_range, sensor_detect_Pr, node_from_XY(start_x, start_y, grid), grid))

            # update nodes_to_visit for start node
            nodes_to_search = update_node_visited_array(nodes_to_search, blue_agents[j])

            # if random <= epsilon then explore randomly and do not use hueristic
            if (random.random() <= epsilon):
                
                blue_agents[j] = find_path(blue_agents[j], grid, nodes_to_search, algorithm_options[0])
            
            else:
                
                blue_agents[j] = find_path(blue_agents[j], grid, nodes_to_search, algorithm_options[2])
            
            scores.append(score_path(blue_agents[j]))

        # leave surivive as is

        # if mutate cohort
        if (j > survive_number - 1) and (j <= (survive_number + mutate_number - 1)) and (i > 0):
                
                blue_agents[j] = mutation(blue_agents[j])
                
                nodes_to_search = update_node_visited_array(nodes_to_search, blue_agents[j])

                # if random <= epsilon then explore randomly and do not use hueristic
                if (random.random() <= epsilon):
                    
                    blue_agents[j] = find_path(blue_agents[j], grid, nodes_to_search, algorithm_options[0])
                
                else:
                    
                    blue_agents[j] = find_path(blue_agents[j], grid, nodes_to_search, algorithm_options[2])

                scores.append(score_path(blue_agents[j]))

    score_list = list(zip(scores, blue_agents))

    # Sort the list in descending order by the values in the first element of each tuple
    sorted_score_list = sorted(score_list, key=lambda x: x[0], reverse=True)

    # reset agents unless part of survive or mutate cohort 
    num_reset = len(blue_agents) - (survive_number + mutate_number)
    temp_agents = []
    
    temp_agents = [t[1] for t in sorted_score_list]
    temp_scores = [t[0] for t in sorted_score_list]
    
    # reset blue_agents
    blue_agents = []
    scores = []
    
    if (i < (epochs-1)):
        # keep survive and mutate
        blue_agents = copy.deepcopy(temp_agents[:-num_reset])
        # keep scores for survive
        scores = copy.deepcopy(temp_scores[:-(num_reset + mutate_number)])
    else:
        # finshed and save final generation and scores in order
        blue_agents = copy.deepcopy(temp_agents)
        # keep scores for survive
        scores = copy.deepcopy(temp_scores)
        
# show results best 10 or num_agents

if num_agents > 10:
    n_a = 10
else:
    n_a = num_agents - 1

best_agents = n_a

if best_agents > num_agents:
    best_agents = num_agents

for a in range(best_agents):
    summary_results(blue_agents[a])
    

