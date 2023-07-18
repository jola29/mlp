from mlp_model import *
from generate_data import *

N = 10000

number_of_training_data = int((3*N)/4)
number_of_test_data = int(N/4)


x_train = random_vectors(number_of_training_data)
y_train = torch.tensor([test_if_in_circle(vect) for vect in x_train],requires_grad=True)


# Create an instance of the model
model = NeuralNetwork()

# Define the loss function and optimizer
loss_function = nn.MSELoss(reduction='mean') #nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)






num_epochs = 10

for epoch in range(num_epochs):
    running_loss = 0.0
    
    optimizer.zero_grad()

    # Forward pass
    outputs = model(x_train)
    #print(outputs)
    #print(y_train)
    loss = loss_function(outputs, y_train)
    # Backward pass and optimization
    loss.backward()
    optimizer.step()

    running_loss += loss.item()

    print(f"Epoch {epoch + 1}, Loss: {running_loss / len(x_train)}")
print("Finished Training")






correct = 0
total = 0

with torch.no_grad(): #stops doing gradient computation, i.e. training of the model
    x_test = random_vectors(number_of_test_data)
    y_test = torch.tensor([test_if_in_circle(vect) for vect in x_test],requires_grad=True)

    outputs = model(x_test)
    _, pred = torch.max(outputs.data, 1)#gives index of maximum value
    predicted = torch.tensor([[(1.0+i)%2 ,(0.0+i)%2] for i in pred])#use those indices to create a vecor similar to those in y_test 
    total = y_test.size(0)
    correct = len([predicted[i] == y_test[i] for i in range(len(predicted))])

print(f"Accuracy: {100 * correct / total}%")


torch.save(model.state_dict(), 'trained_model.pth')#save the model





