from torchvision import datasets, transforms
import torch
import torch.optim as optim
import torch.nn.functional as F
from torchsummary import summary
from tqdm import tqdm


import matplotlib.pyplot as plt

class CustomDataset:
    def __init__(self, dataset_name, train=True, valid=True, batch_size=512):
        self.dataset_name = dataset_name
        self.train = train
        self.valid = valid
        self.batch_size = batch_size
        
        self._load_data()
        self._create_loaders()
    
    def _load_data(self):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        if self.dataset_name == 'MNIST':
            self.train_data = datasets.MNIST('../data', train=True, download=True, transform=transform)
            self.test_data = datasets.MNIST('../data', train=False, download=True, transform=transform)
        elif self.dataset_name == 'CIFAR10':
            self.train_data = datasets.CIFAR10('../data', train=True, download=True, transform=transform)
            self.test_data = datasets.CIFAR10('../data', train=False, download=True, transform=transform)
        # Add more dataset options here
        
    def _create_loaders(self):
        kwargs = {'batch_size': self.batch_size, 'shuffle': True, 'num_workers': 2, 'pin_memory': True}
        
        if self.train:
            self.train_loader = torch.utils.data.DataLoader(self.train_data, **kwargs)
        if self.valid:
            self.valid_loader = torch.utils.data.DataLoader(self.test_data, **kwargs)
    
    def visualize_data(self):
        batch_data, batch_label = next(iter(self.train_loader))
        
        fig = plt.figure()
        for i in range(12):
            plt.subplot(3, 4, i+1)
            plt.tight_layout()
            plt.imshow(batch_data[i].squeeze(0), cmap='gray')
            plt.title(batch_label[i].item())
            plt.xticks([])
            plt.yticks([])
        plt.show()
    
    def get_metadata(self):
        return {
            'dataset_name': self.dataset_name,
            'train': self.train,
            'valid': self.valid,
            'batch_size': self.batch_size,
            'num_classes': len(self.train_data.classes),
            'num_train_samples': len(self.train_data),
            'num_valid_samples': len(self.test_data) if self.valid else 0
        }


import matplotlib.pyplot as plt

class ModelTrainer:
    def __init__(self, model, train_loader, test_loader, device):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.train_losses = []
        self.test_losses = []
        self.train_acc = []
        self.test_acc = []

    def train(self, criterion, optimizer, scheduler, num_epochs):
        self.model.to(self.device)
        
        print('Model Summary:')
        summary(self.model, input_size=(1, 28, 28))

        for epoch in range(1, num_epochs + 1):
            print(f'Epoch {epoch}')
            self._train_epoch(criterion, optimizer)
            self._test_epoch(criterion)
            scheduler.step()

    def _train_epoch(self, criterion, optimizer):
        self.model.train()
        pbar = tqdm(self.train_loader)

        train_loss = 0
        correct = 0
        processed = 0

        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            optimizer.zero_grad()

            # Predict
            pred = self.model(data)

            # Calculate loss
            loss = criterion(pred, target)
            train_loss += loss.item()

            # Backpropagation
            loss.backward()
            optimizer.step()

            correct += self._get_correct_pred_count(pred, target)
            processed += len(data)

            pbar.set_description(
                desc=f'Train: Loss={loss.item():0.4f} Batch_id={batch_idx} Accuracy={100 * correct / processed:0.2f}')

        self.train_acc.append(100 * correct / processed)
        self.train_losses.append(train_loss / len(self.train_loader))

    def _test_epoch(self, criterion):
        self.model.eval()

        test_loss = 0
        correct = 0

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.test_loader):
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                test_loss += criterion(output, target, reduction='sum').item()  # sum up batch loss

                correct += self._get_correct_pred_count(output, target)

        test_loss /= len(self.test_loader.dataset)
        self.test_acc.append(100. * correct / len(self.test_loader.dataset))
        self.test_losses.append(test_loss)

        print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(self.test_loader.dataset),
            100. * correct / len(self.test_loader.dataset)))

    def _get_correct_pred_count(self, prediction, labels):
        return prediction.argmax(dim=1).eq(labels).sum().item()

    def visualize_results(self):
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        axs[0, 0].plot(self.train_losses)
        axs[0, 0].set_title("Training Loss")
        axs[1, 0].plot(self.train_acc)
        axs[1, 0].set_title("Training Accuracy")
        axs[0, 1].plot(self.test_losses)
        axs[0, 1].set_title("Test Loss")
        axs[1, 1].plot(self.test_acc)
        axs[1, 1].set_title("Test Accuracy")
        plt.show()

