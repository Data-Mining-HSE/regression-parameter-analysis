import torch


class LinearRegression(torch.nn.Module):
    def __init__(self, inputSize: int, outputSize: int) -> None:
        super().__init__()
        self.fc = torch.nn.Linear(inputSize, outputSize)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.fc(x)
        return out
