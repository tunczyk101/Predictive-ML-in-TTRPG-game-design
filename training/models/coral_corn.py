import numpy as np
import torch
from coral_pytorch.dataset import (
    corn_label_from_logits,
    levels_from_labelbatch,
    proba_to_label,
)
from coral_pytorch.layers import CoralLayer
from coral_pytorch.losses import coral_loss, corn_loss
from sklearn.base import BaseEstimator, ClassifierMixin
from skorch import NeuralNet
from skorch.dataset import Dataset as SkorchDataset
from torch import nn, sigmoid
from torch.nn import CrossEntropyLoss
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset

from training.constants import NUM_CLASSES, RANDOM_STATE


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
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.num_classes = num_classes
        self.fc = CoralLayer(size_in=64, num_classes=num_classes)

    def forward(self, x):
        x = self.network(x)

        logits = self.fc(x)
        probas = torch.sigmoid(logits)

        return logits, probas

    def predict_proba(self, x):
        return sigmoid(self(x))

    def predict(self, x, threshold: float = 0.5):
        y_pred_score = self.predict_proba(x)
        return (y_pred_score >= threshold).to(torch.int32)


class Coral(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        input_size: int = 53,
        num_classes: int = NUM_CLASSES,
        lambda_reg: float = 0.01,
        learning_rate: float = 0.05,
        num_epochs: int = 100,
        batch_size: int = 128,
    ):
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
            num_workers=0,  # number processes/CPUs to use
        )

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


class SkorchCORAL(NeuralNet):
    def __init__(
        self,
        input_size: int = 30,
        num_classes: int = NUM_CLASSES,
        optimizer__weight_decay: float = 0.01,
        optimizer__lr: float = 0.05,
        optimizer: Optimizer = torch.optim.AdamW,
        *args,
        **kwargs
    ):
        if "module" not in kwargs:
            kwargs["module"] = CORAL_MLP(input_size=input_size, num_classes=num_classes)
        # not in use but GridSearch pass it and it has to be passed for NeuralNet
        if "criterion" not in kwargs:
            kwargs["criterion"] = CrossEntropyLoss
        super().__init__(
            *args,
            optimizer__lr=optimizer__lr,
            optimizer__weight_decay=optimizer__weight_decay,
            optimizer=optimizer,
            **kwargs
        )

    def get_loss(self, y_pred, y_true, X=None, training=False):
        levels = levels_from_labelbatch(y_true, num_classes=self.module_.num_classes)
        logits, _ = y_pred
        return coral_loss(logits, levels.to(DEVICE))

    def fit(self, X, y=None, **fit_params):
        train_dataset = SkorchDataset(X.to_numpy().astype(np.float32), y.to_numpy())
        super().fit(train_dataset.X, train_dataset.y, **fit_params)

    def predict(self, X):
        features = torch.from_numpy(X.values).float()
        features = features.to(DEVICE)

        logits, probas = self.module_(features)
        predicted_labels = proba_to_label(probas).float()

        return predicted_labels.numpy()


class CORN_MLP(nn.Module):
    def __init__(self, input_size: int, num_classes: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            ### Specify CORN layer
            torch.nn.Linear(64, num_classes - 1),
        )

    def forward(self, x):
        return self.network(x)


class Corn(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        input_size: int = 53,
        num_classes: int = NUM_CLASSES,
        lambda_reg: float = 0.01,
        learning_rate: float = 0.05,
        num_epochs: int = 100,
        batch_size: int = 128,
    ):
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
        torch.manual_seed(RANDOM_STATE)
        train_dataset = OrdinalDataset(X.to_numpy(), y.to_numpy())
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=self.batch_size,
            shuffle=True,  # want to shuffle the dataset
            num_workers=0,  # number processes/CPUs to use
        )

        for epoch in range(self.num_epochs):

            self.model = self.model.train()
            for batch_idx, (features, class_labels) in enumerate(train_loader):
                class_labels = class_labels.to(DEVICE)
                features = features.to(DEVICE)
                logits = self.model(features)

                loss = corn_loss(logits, class_labels, self.num_classes)

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
