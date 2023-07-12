#build a simple mlp with 2 neurons in the activation layer, 1 hidden layer and 2 nodes in the output layer

#see https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html 
#for a tutorial on using pytorch to build a neural network

import torch
from torch import nn
from generate_data import *


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(2, 1),
            nn.ReLU(),
            nn.Linear(1, 1),
            nn.ReLU(),
            nn.Linear(1, 2),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

if __name__ == '__main__':
    model = NeuralNetwork()
    print(model)

    X = random_vectors(10)

    
    #X = torch.rand(1, 28, 28, device=device)
    logits = model(X)
    pred_probab = nn.Softmax(dim=1)(logits)
    y_pred = pred_probab.argmax(1)
    print(f"Predicted class: {y_pred}")
    
