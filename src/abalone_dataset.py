from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset


class AbaloneDataset(Dataset):
 
  def __init__(self,file_name: Path, is_train: bool = True) -> None:
    frame = pd.read_csv(file_name)

    train_selection = int(0.9 * len(frame))
 
    if is_train:
        x_selected = frame.iloc[:train_selection, 1:8].values
        y_selected = frame.iloc[:train_selection, 8].values
    else:
        x_selected = frame.iloc[train_selection + 1:, 1:8].values
        y_selected = frame.iloc[train_selection + 1:, 8].values
 
    self.x_selection = torch.tensor(x_selected, dtype=torch.float32)
    self.y_selection = torch.tensor(y_selected, dtype=torch.float32).reshape(-1, 1)

  @property
  def params(self) -> tuple[int, int]:
    return self.x_selection.shape[1], self.y_selection.shape[1]
 
  def __len__(self) -> int:
    return len(self.x_selection)
   
  def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
    return self.x_selection[idx], self.y_selection[idx]
