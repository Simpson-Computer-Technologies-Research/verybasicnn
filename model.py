import torch
import torch.nn as nn
from typing import Any

# A model that predicts whether the provided weight and height is either
# a man or a woman, a binary classification problem.
class Model(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(Model, self).__init__()
        self.ltsm = nn.LSTM(in_channels, out_channels)
        self.relu = nn.ReLU()
        self.linear = nn.Linear(out_channels, out_channels)
        
    
    # Forward propagation
    def forward(self, x: Any):
        x, _ = self.ltsm(x)
        x = self.relu(x)
        x = self.linear(x)
        return x
    
    
    # Train the model
    def train(self, dataset: Any, epochs: int):
        # Define the loss function
        criterion = nn.BCEWithLogitsLoss()
        
        # Define the optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr = 0.001)
        
        # Loop through the epochs
        for epoch in range(epochs):
            # Loop through the dataset
            for x, y in dataset:
                # Forward propagation
                y_pred = self.forward(x)

                # Calculate the loss
                loss = criterion(y_pred, y)
                
                # Backward propagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
                # Print the loss
                print(f"Epoch {epoch + 1} / {epochs} | Loss: {loss.item()}")
    
    
    # Save the model
    def save(self):
        torch.save(self.state_dict(), "model.pth")
    