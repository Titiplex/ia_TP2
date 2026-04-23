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
            activation2,
            output_activation=None):
        super(FashionMLP, self).__init__()

        # TODO
        # Créer un réseau de neurones MLP selon les paramètres d'entrée. Utilisez les couches de torch.nn

        self.flatten = nn.Flatten()  # Couche pour adapter l'entrée 28x28 en vecteur
        self.activation1 = activation1
        self.activation2 = activation2

        self.fc1 = nn.Linear(input_size, h1_size)
        self.fc2 = None

        if h2_size > 0:
            self.fc2 = nn.Linear(h1_size, h2_size)
            self.out = nn.Linear(h2_size, output_size)
        else:
            self.out = nn.Linear(h1_size, output_size)

        self.output_activation = output_activation

    def forward(self, x):
        # TODO
        # Implémentez le forward pass de votre réseau de neurones
        x = self.flatten(x)
        x = self.activation1(self.fc1(x))

        if self.fc2 is not None:
            x = self.activation2(self.fc2(x))

        x = self.out(x)

        if self.output_activation is not None:
            x = self.output_activation(x)

        return x
