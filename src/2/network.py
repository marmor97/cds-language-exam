''' Assignment description - Creating reusable network analysis pipeline
This assignment attempts to create a network pipeline as a reusable command-line tool. This command-line tool will take a given dataset and perform simple network analysis. In particular, it will build networks based on entities appearing together in the same documents.
'''

# System tools
import os
import argparse 
from pathlib import Path

# Data analysis
import pandas as pd
from collections import Counter
from itertools import combinations 

# NLP
import scipy

# Network tools
import networkx as nx
import matplotlib.pyplot as plt
# Setting frame so network isn't too small or big
plt.rcParams["figure.figsize"] = (20,20)


class network_pipeline:
    def __init__(self, args):
        '''
        Inits pipeline by loading an edgelist from the specified path 
        '''
        self.args = args

        print('[INFO] Finding edgelist...')
        self.edgelist = pd.read_csv(self.args["edgelist_name"])
    
    # Function for making a network from the edgelist dataframe
    def make_network(self):
        '''
        Function creating the network using networkx with weights above the argument weight_threshold
        '''
        print('[INFO] Creating network...')

        filtered = self.edgelist[self.edgelist["weights"] > self.args["weight_treshold"]]
        self.network = nx.from_pandas_edgelist(filtered, 'nodeA', 'nodeB', ["weights"])
    
    def plot_network(self):
        '''
        Function that creates a plot of the network and saves it in the specified output folder
        '''
        print('[INFO] Plotting network...')
        # Plotting and saving the image 
        pos = nx.drawing.nx_pylab.draw_spring(self.network, node_size=3.5, with_labels=False)
        nx.draw(self.network, pos, with_labels=True, node_size=25, font_size=5, font_color = "darkblue")
        plt.savefig(os.path.join(self.args["outpath"],"network.png"), dpi=300, bbox_inches="tight")
        print(f'[INFO] network visualization saved in {self.args["outpath"]} as network.png')
        
    def calc_measures(self):
        '''
        Function providing calculations of eigenvector centrality and betweenness centrality and saves this information in the folder specified with the argument outpath
        '''
        # Using nx to calculate eigenvector and betweenness centrality
        ev = nx.eigenvector_centrality(self.network)
        bc = nx.betweenness_centrality(self.network)
        
        # Converting to dataframe
        d = pd.DataFrame({'eigenvector':ev, 'betweenness':bc})
        d.reset_index(level=0, inplace=True)
        
        # Print top scores   
        top_eig = d.sort_values("eigenvector", ascending = False, ignore_index = True)["index"]
        top_bet = d.sort_values("betweenness", ascending = False, ignore_index = True)["index"]

        print("Top 5 nodes with highest eigenvector centrality values:")
        print(top_eig.head(5))
        print()        
        print("Top 5 nodes with highest betweenness centrality values:")
        print(top_bet.head(5))

        # Save as csv
        d.to_csv(os.path.join(self.args["outpath"], "network_info.csv"),index=False)
        print(f'[INFO] Eigenvector and betweeness centrality saved in {self.args["outpath"]} as network_info.csv')

        
def main(): 
    # Add description
    ap = argparse.ArgumentParser(description = "[INFO] creating network pipeline") # Defining an argument parse

    ap.add_argument("-e","--edgelist_name", 
                    required=False, # As I have provided a default name it is not required
                    type = str, # Str type
                    default = os.path.join("..","..","data", "2", "edgelist.csv"), # Setting default to the name of my own edgelist
                    help = "str of edgelist filename")
    
    
    ap.add_argument("-w","--weight_treshold", 
                    required=False, 
                    type = int, 
                    default = 800, 
                    help = "int of threshold weights")
    
    ap.add_argument("-o","--outpath", 
                    required=False, 
                    type = str, 
                    default = os.path.join("..", "..","out", "2"),
                    help = "str of output path")
    
    args = vars(ap.parse_args()) # Adding them together
    
    # Defining network pipeline and gathering edgelist
    network_generation = network_pipeline(args = args) 
        
    # Taking the make_network function to make a network
    network = network_generation.make_network()
    
    # Plot network
    network_generation.plot_network()
    
    # Calculate betweenness and eigenvector centrality
    network_generation.calc_measures()
    
if __name__ == "__main__":
    main()
         