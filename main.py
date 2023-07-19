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
batch_size = 100

for epoch in range(num_epochs):    
    running_loss = 0.0
    # Create batches from the dataset
    for i in range(0, number_of_training_data, batch_size):
        inputs = x_train[i:i + batch_size]
        labels = y_train[i:i + batch_size]
    
    
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = loss_function(outputs, labels)
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
    correct,counter = 0,0
    for elem in predicted:
        if torch.equal(elem, y_test[counter]):
            correct += 1
        counter += 1
    
print(f"Accuracy: {100 * correct / total}%")
#angeblich 100% mit ReLU aber problem für punkte mit großen x und y koordinaten
#probiere mal softmax

#torch.save(model.state_dict(), 'trained_model.pth')#save the model
#with torch.no_grad():
#    x_1 = 0.99999999
#    x_2 = 0.99999999
#    x = torch.tensor([x_1,x_2])
#    failvector = x.unsqueeze(0)

#    out = model(failvector)
#    print(f"Output: {out}")
#    _, index = torch.max(out,1)
#    prediction = torch.tensor([(1.0+index)%2 ,(0.0+index)%2])
#    print(f"Prediction: {prediction}")
#    print(f"truth: {test_if_in_circle(x)}")






