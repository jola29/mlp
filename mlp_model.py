#build a simple mlp with 2 neurons in the activation layer, 1 hidden layer and 2 nodes in the output layer

#see https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html 
#and https://pytorch.org/tutorials/beginner/pytorch_with_examples.html

import torch
from torch import nn
from torch import optim
from generate_data import *


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(2, 1),
            nn.ReLU(),
            nn.Linear(1, 1),
            nn.ReLU(),
            nn.Linear(1, 2),
        )

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = self.linear_relu_stack(x)
        return x

if __name__ == '__main__':
    
    model = NeuralNetwork()
    #print(model)
    X = random_vectors(5)
    Y = torch.tensor([test_if_in_circle(vect) for vect in X])
    print(X)
    print(Y)

    

   
