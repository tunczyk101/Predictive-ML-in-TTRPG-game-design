from coral_pytorch.layers import CoralLayer
from torch import int32, nn, sigmoid


class CoralMLP(nn.Module):
    def __init__(
        self,
        input_size: int,
        num_classes: int,
        num_hidden_1: int = 100,
        num_hidden_2: int = 50,
    ):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, num_hidden_1),
            nn.ReLU(),
            nn.Linear(num_hidden_1, num_hidden_2),
            nn.ReLU(),
        )
        self.fc = CoralLayer(size_in=num_hidden_2, num_classes=num_classes)

    def forward(self, x):
        x = self.network(x)

        logits = self.fc(x)
        probas = sigmoid(logits)

        return logits, probas

    def predict_proba(self, x):
        return sigmoid(self(x))

    def predict(self, x, threshold: float = 0.5):
        y_pred_score = self.predict_proba(x)
        return (y_pred_score > threshold).to(int32)
