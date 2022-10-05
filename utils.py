import torch
from IPython.display import display, Audio
import numpy as np

def from_numpy(x):
    return torch.from_numpy(x).float()

def to_numpy(tensor):
    return tensor.to('cpu').detach().numpy()

def normalize(x):
    return (x - np.mean(x))/(np.std(x) + 1e-8)

def play(sound: np.ndarray, rate = 16000):
    display(Audio(sound, rate = rate))

