import torch
import torch.nn as nn
import pandas as pd

# Load the dataset
dataset = pd.read_csv('data/pong_data.csv')
print(dataset.head())


class PongAI(nn.Module):
    def __init__(self):
        print("Initializing PongAI model")
        super(PongAI, self).__init__()
    
    def forward(self, x):
        # Define the forward pass
        # For now, just return the input
        return x