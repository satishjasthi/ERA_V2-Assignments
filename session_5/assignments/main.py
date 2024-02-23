import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from model import Net
from utils import CustomDataset, ModelTrainer

# Check if CUDA is available
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print("CUDA Available?", use_cuda)

# Create data handler
data_handler = CustomDataset('MNIST',
                            train=True,
                            valid=True, 
                            batch_size=512)

# Visualize the data
data_handler.visualize_data()

# Create the network
network = Net()

# Create the model trainer
model_handler = ModelTrainer(model=network,
                             train_loader=data_handler.train_loader,
                             test_loader=data_handler.valid_loader,
                             device=device)

# Define the optimizer
optimizer = optim.SGD(network.parameters(), lr=0.01, momentum=0.9)

# Define the learning rate scheduler
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1, verbose=True)

# Define the loss function
criterion = F.nll_loss

# Set the number of epochs
num_epochs = 20

# Train the model
model_handler.train(criterion=criterion, optimizer=optimizer, scheduler=scheduler, num_epochs=num_epochs)

# Visualize the results
model_handler.visualize_results()