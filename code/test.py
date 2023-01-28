from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import snntorch as snn
import torch
import mlflow
import logging

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


def load_data():
    data_path='/home/maria/Documents/MNISTdata'
    # Define a transform
    transform = transforms.Compose([
                transforms.Resize((28,28)),
                transforms.Grayscale(),
                transforms.ToTensor(),
                transforms.Normalize((0,), (1,))])

    mnist_train = datasets.MNIST(data_path, train=True, download=True, transform=transform)


    train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True)

#How to log params
def make_params():
    # Network Architecture
    num_inputs = 28*28
    num_hidden = 1000
    num_outputs = 10

    # Temporal Dynamics
    num_steps = 25
    beta = 0.95
    params={'num_inputs':num_inputs,
            'num_hidden':num_hidden, num_steps':num_steps',
            num_outputs':num_outputs', 'beta':beta}
    return params


# Define Network
class Net(nn.Module):
    def __init__(self, params):
        super().__init__()


        # Initialize layers
        self.fc1 = nn.Linear(params['num_inputs'], params['num_hidden'])
        self.lif1 = snn.Leaky(beta=params['beta'])
        self.fc2 = nn.Linear(params['num_hidden'], params['num_outputs'])
        self.lif2 = snn.Leaky(beta=params['beta'])

    def forward(self, x):

        # Initialize hidden states at t=0
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        # Record the final layer
        spk2_rec = []
        mem2_rec = []

        for step in range(num_steps):
            cur1 = self.fc1(x)
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            spk2_rec.append(spk2)
            mem2_rec.append(mem2)

        return torch.stack(spk2_rec, dim=0), torch.stack(mem2_rec, dim=0)

# Load the network onto CUDA if available
net = Net()

