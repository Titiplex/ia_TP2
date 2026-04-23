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
        layers = [nn.Linear(input_size, h1_size), activation1]

        if h2_size > 0:
            layers.append(nn.Linear(h1_size, h2_size))
            layers.append(activation2)
            layers.append(nn.Linear(h2_size, output_size))
        else:
            layers.append(nn.Linear(h1_size, output_size))

        if output_activation is not None:
            layers.append(output_activation)

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        # TODO
        # Implémentez le forward pass de votre réseau de neurones
        x = self.flatten(x)
        x = self.network(x)
        return x
