import torch

from dataset_dataloader import load_MNIST_into_dataloaders
from model import FirstModel

### Dataset download, split, and create DataLoaders 
train_dl, valid_dl, test_dl = load_MNIST_into_dataloaders(valid_split=0.3, root='test', verbose=True)

### Create neural network with 3 layers
model = FirstModel(28*28, 512,10)

### Set Hyperparameters
lr = 1e-3
epochs = 10

### set loss function and optimizer
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

### Training loop
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train() # Set for training mode

    for batch, (X, y) in enumerate(dataloader):
        # Forward Pass
        pred = model.forward(X) # Make predictions using X from batch
        loss = loss_fn(pred, y) # Calculate loss using prediction and label

        # Backpropagation
        loss.backward() # computes derivative of loss with respect to each parameter 
        optimizer.step() # updates the values of each parameter
        optimizer.zero_grad() # clears grads for every parameter

        # Print loss every 100 batches
        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

### Testing loop
def test_loop(dataloader, model, loss_fn):
    model.eval() # Set for evaluation mode
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad(): # No need to compute gradients
        for X, y in dataloader:
            pred = model.forward(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

for epoch in range(epochs):
    print(f'Epoch {epoch+1}\n-------------------------------')
    train_loop(train_dl, model, loss_fn, optimizer)

print('Training done\n')
test_loop(valid_dl, model, loss_fn)
print('Validation done\n')
test_loop(test_dl, model, loss_fn)
print('Validation done')  
