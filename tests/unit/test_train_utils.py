import math

import torch
from torch.utils.data import DataLoader, TensorDataset

from src.models.cafn import CAFN
from src.train import evaluate, train_val_split


class TestTrainValSplit:
    def test_normal_split_partitions_dataset(self):
        ds = TensorDataset(torch.arange(10))
        train_ds, val_ds = train_val_split(ds, val_split=0.2, seed=0)
        assert len(train_ds) + len(val_ds) == 10
        assert len(val_ds) == 2

    def test_single_sample_dataset_falls_back_to_all_train(self):
        ds = TensorDataset(torch.arange(1))
        train_ds, val_ds = train_val_split(ds, val_split=0.2, seed=0)
        assert len(train_ds) == 1
        assert len(val_ds) == 0

    def test_split_is_deterministic_for_fixed_seed(self):
        ds = TensorDataset(torch.arange(20))
        train_a, val_a = train_val_split(ds, val_split=0.3, seed=5)
        train_b, val_b = train_val_split(ds, val_split=0.3, seed=5)
        assert list(train_a.indices) == list(train_b.indices)
        assert list(val_a.indices) == list(val_b.indices)


class TestEvaluate:
    def test_empty_loader_returns_nan(self, tiny_config):
        model = CAFN(tiny_config.model)
        empty_ds = TensorDataset(torch.empty(0))
        loader = DataLoader(empty_ds, batch_size=4)
        metrics = evaluate(model, loader, torch.device("cpu"))
        assert math.isnan(metrics["loss"])
        assert math.isnan(metrics["accuracy"])
