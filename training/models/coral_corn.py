import numpy as np
import pandas as pd
import torch
from coral_pytorch.dataset import (
    corn_label_from_logits,
    levels_from_labelbatch,
    proba_to_label,
)
from coral_pytorch.layers import CoralLayer
from coral_pytorch.losses import coral_loss, corn_loss
from sklearn.base import BaseEstimator, ClassifierMixin
from torch import nn, sigmoid
from torch.utils.data import DataLoader, Dataset

from training.constants import RANDOM_STATE


NUM_CLASSES = 23
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(RANDOM_STATE)


class OrdinalDataset(Dataset):
    def __init__(self, feature_array, label_array, dtype=np.float32):
        self.features = feature_array.astype(np.float32)
        self.labels = label_array

    def __getitem__(self, index):
        inputs = self.features[index]
        label = self.labels[index]
        return inputs, label

    def __len__(self):
        return self.labels.shape[0]


class CORAL_MLP(nn.Module):
    def __init__(self, input_size: int, num_classes: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 100),
            nn.ReLU(),
            nn.Linear(100, 50),
            nn.ReLU(),
        )
        self.fc = CoralLayer(size_in=50, num_classes=num_classes)

    def forward(self, x):
        x = self.network(x)

        logits = self.fc(x)
        probas = torch.sigmoid(logits)

        return logits, probas

    def predict_proba(self, x):
        return sigmoid(self(x))

    def predict(self, x, threshold: float = 0.5):
        y_pred_score = self.predict_proba(x)
        return (y_pred_score > threshold).to(torch.int32)


class Coral(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        input_size=53,
        num_classes=NUM_CLASSES,
        lambda_reg=0.01,
        learning_rate=0.05,
        num_epochs=100,
        batch_size=128,
    ):
        torch.manual_seed(RANDOM_STATE)
        self.input_size = input_size
        self.lambda_reg = lambda_reg
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.model = CORAL_MLP(input_size=self.input_size, num_classes=num_classes)
        self.model.to(DEVICE)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=learning_rate, weight_decay=self.lambda_reg
        )

    def fit(self, X, y):
        train_dataset = OrdinalDataset(X.to_numpy(), y.to_numpy())
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=self.batch_size,
            shuffle=True,  # want to shuffle the dataset
            num_workers=0,
        )  # number processes/CPUs to use

        for epoch in range(self.num_epochs):

            self.model = self.model.train()
            for batch_idx, (features, class_labels) in enumerate(train_loader):

                levels = levels_from_labelbatch(
                    class_labels, num_classes=self.num_classes
                )

                features = features.to(DEVICE)
                levels = levels.to(DEVICE)
                logits, probas = self.model(features)

                loss = coral_loss(logits, levels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # if not batch_idx % 200:
                #     print(
                #         "Epoch: %03d/%03d | Batch %03d/%03d | Loss: %.4f"
                #         % (epoch + 1, self.num_epochs, batch_idx, len(train_loader), loss)
                #     )

    def predict(self, X):
        features = torch.from_numpy(X.values).float()
        features = features.to(DEVICE)

        logits, probas = self.model(features)
        predicted_labels = proba_to_label(probas).float()

        return predicted_labels.numpy()


class CORN_MLP(nn.Module):
    def __init__(self, input_size: int, num_classes: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 100),
            nn.ReLU(),
            nn.Linear(100, 50),
            nn.ReLU(),
            ### Specify CORN layer
            torch.nn.Linear(50, (num_classes - 1)),
        )

    def forward(self, x):
        logits = self.network(x)

        return logits


class Corn(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        input_size=53,
        num_classes=NUM_CLASSES,
        lambda_reg=0.01,
        learning_rate=0.05,
        num_epochs=100,
        batch_size=128,
    ):
        torch.manual_seed(RANDOM_STATE)
        self.learning_rate = learning_rate
        self.input_size = input_size
        self.lambda_reg = lambda_reg
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.model = CORN_MLP(input_size=input_size, num_classes=num_classes)
        self.model.to(DEVICE)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=learning_rate, weight_decay=lambda_reg
        )

    def fit(self, X, y):
        train_dataset = OrdinalDataset(X.to_numpy(), y.to_numpy())
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=self.batch_size,
            shuffle=True,  # want to shuffle the dataset
            num_workers=0,
        )  # number processes/CPUs to use

        for epoch in range(self.num_epochs):

            self.model = self.model.train()
            for batch_idx, (features, class_labels) in enumerate(train_loader):

                class_labels = class_labels.to(DEVICE)
                features = features.to(DEVICE)
                logits = self.model(features)

                #### CORN loss
                loss = corn_loss(logits, class_labels, self.num_classes)
                ###--------------------------------------------------------------------###

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # if not batch_idx % 200:
                #     print('Epoch: %03d/%03d | Batch %03d/%03d | Cost: %.4f'
                #           % (epoch + 1, self.num_epochs, batch_idx,
                #              len(train_loader), loss))

    def predict(self, X):
        features = torch.from_numpy(X.values).float()
        features = features.to(DEVICE)

        logits = self.model(features)
        predicted_labels = corn_label_from_logits(logits).float()

        return predicted_labels
