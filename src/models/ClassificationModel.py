from abc import ABC, abstractmethod
import torch

class ClassificationModel(ABC,torch.nn.Module):

    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self):
        pass

    @torch.no_grad()
    def predict(self, logprobabilities):
        return  logprobabilities.argmax(-1)

    @abstractmethod
    def save(self, path="model.pth",**kwargs):
        pass

    @abstractmethod
    def load(self, path):
        pass