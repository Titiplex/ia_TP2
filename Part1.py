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
        return activation(np.dot(w, x) + b)


class Layer():
    def __init__(self, input_size, output_size, input_structure):
        # TODO initialiser la matrice de poids et le vecteur de biais avec des valeurs appropriées
        self.weights = np.random.randn(output_size, input_size) * 0.01
        self.biases = np.zeros(output_size)
        self.input_structure = input_structure

    def __call__(self, x, activation):
        # TODO calculez le vecteur de sortie de la couche
        x = np.array(x).reshape(-1)
        return activation(np.dot(self.weights, x) + self.biases)


# Fonctions d'activation
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def tanh(x):
    # TODO
    return np.tanh(x)


def leaky_ReLu(x, alpha=0.01):
    # TODO
    return np.where(x > 0, x, alpha * x)  # where more accurate considering the vector/matrix character of x


def softmax(x):
    # TODO
    return np.exp(x - np.max(x)) / np.sum(np.exp(x - np.max(x)))


# Fonctions de perte
def MAE(y_true, y_pred):
    # TODO
    return np.mean(np.abs(y_true - y_pred))


def MSE(y_true, y_pred):
    # TODO
    return np.mean(np.pow(y_true - y_pred, 2))


def log_cosh(y_true, y_pred):
    # TODO ln(cosh(x))
    return np.mean(np.log(np.cosh(y_pred - y_true)))


def cross_entropy(y_true, y_pred):
    # TODO N'oubliez pas d'éviter la division par 0 avec un epsilon (1e-10). Assumez que y_true est un vecteur one-hot
    epsilon = 1e-10
    return - np.sum(y_true * np.log(y_pred + epsilon))


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

            # TODO
            # Initialiser les gradients à 0
            # Prédire le minibatch
            # calculer la perte
            # Avancer
            # Calculer les métriques d'évaluation

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

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

                # TODO
                # Prédire les classes
                # Calculer la perte
                # Calculer les métriques d'évaluation
                outputs = model(images)

                loss = criterion(outputs, labels)
                running_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_test_loss = running_loss / len(test_loader)
        test_losses.append(avg_test_loss)
        test_accuracy = 100 * correct / total
        test_accuracies.append(test_accuracy)

        print(
            f"Epoch [{epoch + 1}/{epochs}] | Train Loss: {avg_train_loss:.4f} | Test Loss: {avg_test_loss:.4f} | Train Accuracy: {train_accuracy:.2f} | Test Accuracy: {test_accuracy:.2f}%")
    return train_losses, test_losses, train_accuracies, test_accuracies
