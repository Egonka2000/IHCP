import torch
import torch.backends
import torch.backends.mps
import torch.optim as optim
from torch import nn, Tensor
from torch.utils.data import DataLoader

class NeuralNetwork():
    def __init__(self):
        self.device = ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        
    def build_network(self, input_size: int, hidden_layers: list[int], output_size: int, learning_rate: float):
        neuron_counts = [input_size, *hidden_layers, output_size]
        layers = []
        for i in range(len(neuron_counts) - 1):
            layers.append(nn.Linear(neuron_counts[i], neuron_counts[i + 1]))
            if i != len(neuron_counts) - 2:
                layers.append(nn.ReLU())
        self.model = nn.Sequential(*layers).to(self.device)
        self.loss = nn.MSELoss()
        self.optimizer = optim.Adam(params=self.model.parameters(), lr=learning_rate)
        #self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.8)

    def train_loop(self, dataloader: DataLoader):
        size = len(dataloader.dataset)
        self.model.train()
        for batch_idx, (X, y) in enumerate(dataloader):
            X, y = X.to(self.device), y.to(self.device)
            
            self.optimizer.zero_grad()

            pred = self.model(X)
            loss = self.loss(pred, y)

            loss.backward()
            self.optimizer.step()
            #self.scheduler.step()

            if batch_idx % 100 == 0:
                curr = (batch_idx + 1) * len(X)
                print(f"loss: {loss.item():.4f} [{curr:>5d}/{size:>5d}]")

    def test_loop(self, dataloader: DataLoader):
        size = len(dataloader.dataset)

        self.model.eval()
        test_loss = 0

        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(self.device), y.to(self.device)
                pred = self.model(X)
                loss = self.loss(pred, y)
                test_loss += loss.item() * X.size(0)

        test_loss /= size
        print(f"Test Error: \n Avg loss: {test_loss:.6f}\n")
    
    def test_one(self, input_data: Tensor):
        self.model.eval()
        with torch.no_grad():
            input_data = input_data.to(self.device)
            pred = self.model(input_data)
            return pred.reshape(1, -1)
    
    def save_model(self, filename: str):
        torch.save(self.model.state_dict(), filename)
        print(f"Saved PyTorch Model State to {filename}")
    
    def load_model(self, filename: str):
        self.model.load_state_dict(torch.load(filename, weights_only=True))
        print(f"Loaded PyTorch Model State from {filename}")