import torch
import torch.nn as nn
import torch.optim as optim
from Networks import CIFAR10_FPN, BasicBlock
from utils import train_class_net, cifar_loaders

device = 'cuda'
print('device = ', device)
seed = 47
torch.manual_seed(seed)
save_dir = "results/"

# -----------------------------------------------------------------------------
# Network setup
# -----------------------------------------------------------------------------
contraction_factor = 0.5
lat_layers = 5
data_layers = 16
num_channels = 35
T = CIFAR10_FPN(lat_layers=lat_layers, num_channels=num_channels,
                contraction_factor=contraction_factor,
                data_layers=data_layers,
                architecture='Explicit').to(device)
eps = 1.0e-1
max_depth = 50
num_classes = 10

# -----------------------------------------------------------------------------
# Training settings
# -----------------------------------------------------------------------------
max_epochs = 10
learning_rate = 1.0e-3
weight_decay = 1e-3
optimizer = optim.Adam(T.parameters(), lr=learning_rate,
                       weight_decay=weight_decay)
lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.5)
checkpt_path = './models/'
criterion = nn.CrossEntropyLoss()

print('weight_decay = ', weight_decay, ', max_depth = ', max_depth, ', eps = ',
      eps, ', learning_rate = ', learning_rate)

# -----------------------------------------------------------------------------
# Load dataset
# -----------------------------------------------------------------------------
batch_size = 100
test_batch_size = 400
train_loader, test_loader = cifar_loaders(train_batch_size=batch_size,
                                          test_batch_size=400, augment=False)

# train network!
T = train_class_net(T, max_epochs, lr_scheduler, train_loader,
                    test_loader, optimizer, criterion, num_classes,
                    eps, max_depth, device, save_dir=save_dir)
