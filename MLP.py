import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np

class FashionMLP(nn.Module):
    def __init__(
            self,
            input_size,
            h1_size,
            h2_size,
            output_size,
            activation1,
            activation2):
        super(FashionMLP, self).__init__()
        
        # TODO
        # Créer un réseau de neurones MLP selon les paramètres d'entrée. Utilisez les couches de torch.nn

        self.flatten = nn.Flatten() # Couche pour adapter l'entrée 28x28 en vecteur

    def forward(self, x):
        # TODO
        # Implémentez le forward pass de votre réseau de neurones
        logits = None
        return logits