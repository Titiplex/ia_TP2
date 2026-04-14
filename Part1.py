import numpy as np
import torch

# Partie de réseaux neuronaux
class Neuron():
    def __init__(self):
        # Le neurone n'a pas d'état propre ici, 
        # les poids et biais sont passé en paramètres de __call__
        pass

    def __call__(self, x, w, b, activation) -> float:
        # x : vecteur d'entrée
        # w : vecteur de poids pour ce neurone
        # b : biais (scalaire) pour ce neurone
        # TODO
        pass

class Layer():
    def __init__(self, input_size, output_size, input_structure):
        # TODO initialiser la matrice de poids et le vecteur de biais avec des valeurs appropriées
        self.weights = None
        self.biases = None

    def __call__(self, x, activation):
        # TODO calculez le vecteur de sortie de la couche
        pass

# Fonctions d'activation
def sigmoid(x):
    # TODO
    pass

def tanh(x):
    # TODO
    pass

def leaky_ReLu(x, alpha=0.01):
    # TODO
    pass

def softmax(x):
    # TODO
    pass

# Fonctions de perte
def MAE(y_true, y_pred):
    # TODO
    pass

def MSE(y_true, y_pred):
    # TODO
    pass

def log_cosh(y_true, y_pred):
    # TODO ln(cosh(x))
    pass

def cross_entropy(y_true, y_pred):
    # TODO N'oubliez pas d'éviter la division par 0 avec un epsilon (1e-10). Assumez que y_true est un vecteur one-hot
    pass

def train(model, epochs, optimizer, criterion, train_loader, test_loader, device):
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []

    print("Starting training...")
    # TODO
    # Implémentez l'algorithme d'entrainement pour toutes les epochs
    for epoch in range(epochs):

        # Entrainement
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            #TODO
            # Initialiser les gradients à 0
            # Prédire le minibatch
            # calculer la perte
            # Avancer
            # Calculer les métriques d'évaluation
            
            outputs = None
            loss = None
            
            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        train_accuracy = 100 * correct / total
        train_accuracies.append(train_accuracy)
        
        # Évaluation
        model.eval()
        correct = 0
        total = 0
        running_loss = 0.0
        
        # Désactiver le gradient pour efficacité
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)

                #TODO
                # Prédire les classes
                # Calculer la perte
                # Calculer les métriques d'évaluation
                outputs = None

                loss =None
                running_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
        avg_test_loss = running_loss / len(test_loader)
        test_losses.append(avg_test_loss)
        test_accuracy = 100 * correct / total
        test_accuracies.append(test_accuracy)
        
        print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {avg_train_loss:.4f} | Test Loss: {avg_test_loss:.4f} | Train Accuracy: {train_accuracy:.2f} | Test Accuracy: {test_accuracy:.2f}%")
    return train_losses, test_losses, train_accuracies, test_accuracies
