import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    # This class defines the structure of the neural network.
    def __init__(self):
        super(Net, self).__init__()
        # Define the convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3)
        # Define the fully connected layers
        self.fc1 = nn.Linear(4096, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        # Apply ReLU activation to the output of the first convolutional layer
        x = F.relu(self.conv1(x), 2)  # 28>26 | 1>3 | 1>1
        # Apply ReLU activation and max pooling to the output of the second convolutional layer
        x = F.relu(F.max_pool2d(self.conv2(x), 2))  # 26>24>12 | 3>5>6 | 1>1>2
        # Apply ReLU activation to the output of the third convolutional layer
        x = F.relu(self.conv3(x), 2)  # 12>10 | 6>10 | 2>2
        # Apply ReLU activation and max pooling to the output of the fourth convolutional layer
        x = F.relu(F.max_pool2d(self.conv4(x), 2))  # 10>8>4 | 10>14>16 | 2>2>4
        # Reshape the output to a 1D tensor
        x = x.view(-1, 4096)  # 4*4*256 = 4096
        # Apply ReLU activation to the output of the first fully connected layer
        x = F.relu(self.fc1(x))
        # Apply the second fully connected layer
        x = self.fc2(x)
        # Apply log softmax activation to the output
        return F.log_softmax(x, dim=1)