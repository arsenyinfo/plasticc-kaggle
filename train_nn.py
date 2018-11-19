from typing import Optional, Callable
import os

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.serialization import save
import numpy as np
import pandas as pd
from glog import logger
import joblib as jl
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from fire import Fire


class Baseline(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.bn = nn.BatchNorm1d(num_features=n_features)
        self.model = nn.Sequential(
            nn.Linear(n_features, 1024),
            nn.ReLU(),
            nn.Dropout(p=.5),
            nn.Linear(1024, 14),
        )

    def forward(self, x):
        x = self.bn(x)
        return self.model(x)


class PlasticcDataset(Dataset):
    def __init__(self, x_data: np.array, y_data: np.array, folds: tuple):
        data = zip(x_data, y_data)
        self.data = [x for i, x in enumerate(data) if i % 5 in folds]
        logger.info(f'There are {len(self.data)} records in the dataset')
        self.features_shape = x_data.shape

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]


def map_classes(y_full):
    classes = sorted(list(set(y_full)))
    mapping = {}
    for i, y in enumerate(classes):
        mapping[y] = i

    logger.info(f'Mapping is {mapping}')
    return np.array([mapping[y] for y in y_full])


def read_train():
    data = pd.read_csv('data/processed_training.csv', engine='c', sep=';')
    y_full = data.pop('target').values
    x_full = data.values.astype('float32')
    x_full[np.isnan(x_full)] = 0
    x_full[np.isinf(x_full)] = 0
    return x_full, y_full


def prepare_data():
    if os.path.exists('data/train.bin'):
        return jl.load('data/train.bin')

    x_full, y_full = read_train()
    imputer = SimpleImputer()
    vt = VarianceThreshold(threshold=.0001)
    pipeline = make_pipeline(imputer, vt, StandardScaler())

    x_full = pipeline.fit_transform(x_full)
    jl.dump(imputer, 'preprocess.bin')
    y_full = map_classes(y_full)
    x_full = x_full.astype('float32')

    jl.dump((x_full, y_full), 'data/train.bin')

    return x_full, y_full


def make_dataloaders():
    x_full, y_full = prepare_data()

    train = PlasticcDataset(x_data=x_full, y_data=y_full, folds=(0, 1, 2, 3))
    val = PlasticcDataset(x_data=x_full, y_data=y_full, folds=(4,))

    shared_params = {'batch_size': 2048, 'shuffle': True}

    train = DataLoader(train, drop_last=True, **shared_params)
    val = DataLoader(val, drop_last=False, **shared_params)
    return train, val


class Trainer:
    def __init__(self,
                 model: nn.Module,
                 train: DataLoader,
                 val: DataLoader,
                 epochs: int = 500,
                 optimizer: Optional[torch.optim.Optimizer] = None,
                 loss_fn: Optional[Callable] = None,
                 scheduler: Optional[ReduceLROnPlateau] = None,
                 reg_lambda: float = .00002,
                 reg_norm: int = 1,
                 device: str = 'cuda:0',
                 checkpoint: str = './model.pt'):
        self.epochs = epochs
        self.model = model.to(device)
        self.device = device
        self.train = train
        self.val = val
        self.optimizer = optimizer if optimizer is not None else torch.optim.Adam(model.parameters(), lr=1e-3)
        self.scheduler = scheduler if scheduler is not None else ReduceLROnPlateau(optimizer=self.optimizer,
                                                                                   verbose=True)
        self.loss_fn = loss_fn if loss_fn is not None else F.cross_entropy
        self.reg_lambda = reg_lambda
        self.reg_norm = reg_norm
        self.current_metric = -np.inf
        self.last_improvement = 0
        self.checkpoint = checkpoint

    def fit_one_epoch(self, n_epoch):
        self.model.train(True)
        losses, reg_losses = [], []

        for i, (x, y) in enumerate(self.train):
            x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(x)

            loss = self.loss_fn(outputs, y)
            losses.append(loss.item())

            for param in self.model.model.parameters():
                loss += self.reg_lambda * torch.norm(param, p=self.reg_norm)

            reg_loss = loss.item()
            reg_losses.append(reg_loss)

            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1)

            self.optimizer.step()

        self.model.train(False)

        val_losses = []
        y_pred_acc, y_true_acc = [], []

        with torch.no_grad():
            for i, (x, y) in enumerate(self.val):
                x, y = x.to(self.device), y.to(self.device)
                outputs = self.model(x)

                loss = self.loss_fn(outputs, y)
                val_losses.append(loss.item())

                y_pred_acc.append(outputs.detach().cpu().numpy())
                y_true_acc.append(y.detach().cpu().numpy())

        train_loss = np.mean(losses)
        train_reg_loss = np.mean(reg_losses)
        val_loss = np.mean(val_losses)
        msg = f'Epoch {n_epoch}: train loss is {train_loss:.5f} (raw), {train_reg_loss:.5f} (reg); val loss is {val_loss:.5f}'
        logger.info(msg)

        self.scheduler.step(metrics=val_loss, epoch=n_epoch)
        y_true_acc, y_pred_acc = map(np.vstack, (y_true_acc, y_pred_acc))

        metric = self.evaluate(y_pred=y_pred_acc, y_true=y_true_acc)

        if metric > self.current_metric:
            self.current_metric = metric
            self.last_improvement = n_epoch
            save(self.model, f=self.checkpoint)
            logger.info(f'Best model has been saved at {n_epoch}, accuracy is {metric:.4f}')

        return train_loss, val_loss, metric

    def evaluate(self, y_pred, y_true):
        return (y_pred.argmax(-1) == y_true).sum() / y_true.shape[-1]

    def fit(self):
        for i in range(self.epochs):
            self.fit_one_epoch(i)


def fit(**kwargs):
    train, val = make_dataloaders()
    trainer = Trainer(model=Baseline(n_features=train.dataset.features_shape[1]),
                      train=train,
                      val=val,
                      loss_fn=F.cross_entropy,
                      **kwargs
                      )
    trainer.fit()


if __name__ == '__main__':
    Fire(fit)
