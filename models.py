import torch.nn as nn

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(64, 64, bias = False),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Linear(64, 256, bias = False), 
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Linear(256, 1024, bias = False),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 4096, bias = False),
            nn.BatchNorm1d(4096),
            nn.LeakyReLU(),
            nn.Linear(4096, 16000, bias = False)
        )
        
    def forward(self, Z):
        return self.main(Z)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(16000, 4096, bias = False),
            nn.BatchNorm1d(4096),
            nn.LeakyReLU(),
            nn.Linear(4096, 1024, bias = False),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 256, bias = False),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Linear(256, 64, bias = False),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Linear(64, 1, bias = False),
            nn.Sigmoid()
        )
            
    def forward(self, X):
        return self.main(X)

class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(16000, 4096, bias = False),
            nn.BatchNorm1d(4096),
            nn.LeakyReLU(),
            nn.Linear(4096, 1024, bias = False),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 256, bias = False),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Linear(256, 64, bias = False),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Linear(64, 1, bias = False),
        )
            
    def forward(self, X):
        return self.main(X)
        