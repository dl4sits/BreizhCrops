from abc import ABC, abstractmethod
import torch
from sklearn.base import BaseEstimator

class ClassificationModel(ABC,torch.nn.Module, BaseEstimator):

    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self):
        pass # return logprobabilities

    @torch.no_grad()
    def predict(self, logprobabilities):
        return  logprobabilities.argmax(-1)

    @abstractmethod
    def save(self, path="model.pth",**kwargs):
        pass

    @abstractmethod
    def load(self, path):
        pass #return snapshot