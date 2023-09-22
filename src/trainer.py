import numpy as np
import torch
from torcheval.metrics.metric import Metric


class Trainer:
    def __init__(self, num_epochs: int, model: torch.nn.Module, criterion: torch.nn.Module,
                optimizer: torch.nn.Module, lr_scheduler: torch.nn.Module, train_loader: torch.utils.data.DataLoader,
                test_loader: torch.utils.data.DataLoader, metric: Metric,
                verbose: bool = False) -> None:
    
        self._num_epochs = num_epochs
        self._model = model
        self._criterion = criterion
        self._lr_scheduler = lr_scheduler
        self._train_loader = train_loader
        self._verbose = verbose
        self._optimizer = optimizer
        self._test_loader = test_loader
        self._metric = metric

    def train(self) -> list[float]:
        self._model.train()
        
        epoch_loss_hist = []
        for epoch in range(self._num_epochs):
            
            loss_hist = []
            for data, labels in self._train_loader:
                
                pred_y = self._model(data)
                loss = self._criterion(pred_y, labels)
        
                self._optimizer.zero_grad()

                if isinstance(self._optimizer, torch.optim.LBFGS):
                    def loss_closure():
                        self._optimizer.zero_grad()
                        output = self._model(data)
                        loss_val = self._criterion(output, labels)
                        loss_val.backward()
                        return loss_val
                    self._optimizer.step(loss_closure)
                else:
                    loss.backward()
                    self._optimizer.step()

                self._lr_scheduler.step()
                loss_hist.append(loss.cpu().detach().item())
            
            epoch_loss = np.mean(loss_hist)
            epoch_loss_hist.append(epoch_loss)
            if self._verbose:
                print(f'epoch {epoch}, loss {epoch_loss}')

        return epoch_loss_hist
    
    @torch.inference_mode()
    def eval(self) -> float:
        self._model.eval()

        for data, labels in self._test_loader:
            pred_y = self._model(data)
            self._metric.update(pred_y, labels)
        return self._metric.compute().cpu().item()
