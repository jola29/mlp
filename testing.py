from mlp_model import *
from generate_data import *


# Create an instance of the model
model = NeuralNetwork()

# Load the saved state dictionary into the model
model.load_state_dict(torch.load('trained_model.pth'))



weights = []
biases = []
# Access the weights and biases of the model's layers
for name, param in model.named_parameters():
    if 'weight' in name:
        #print(f"Layer: {name}, Weights: {param.tolist()}")
        weights += param.view(-1).tolist()
    elif 'bias' in name:
        #print(f"Layer: {name}, Bias: {param.tolist()}")
        biases += param.view(-1).tolist()



x_1 = 0.5
x_2 = 0.5
radius = x_1**2 + x_2**2
vector = torch.tensor([x_1,x_2])

#print(weights)
#print(biases)
node_1 = x_1 * weights[0] + x_2*weights[1] + biases[0]
node_1 = torch.nn.functional.relu(torch.tensor(node_1))
node_2 = node_1 * weights[2] + biases[1]
node_2 = torch.nn.functional.relu(node_2)

output_probabs = torch.tensor([node_2* weights[3] + biases[2], node_2* weights[4] + biases[3]])
#print(output_probabs)
_, index = torch.max(output_probabs,0) #index of the maximum
output = [(1.0+index.item())%2,(0.0+index.item())%2]
#torch.no_grad()
#print(output)
#print(test_if_in_circle(vector))

vector = vector.unsqueeze(0)

out = model(vector)
print(out)
print(output_probabs)


