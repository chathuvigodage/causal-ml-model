import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping

from torch.utils.data import DataLoader
from src.utils.torch import torchDataset


class DRNet(pl.LightningModule):
    def __init__(self, config):
        super(DRNet, self).__init__()

        torch.manual_seed(42)

        # Config
        self.config = config
        self.learningRate = config.get('learningRate')
        self.batch_size = config.get('batchSize')
        self.num_steps = config.get('numSteps')
        self.num_layers = config.get('numLayers')
        self.input_size = config.get('inputSize')
        self.hidden_size = config.get('hiddenSize')
        self.num_heads = config.get('numHeads')

        # --- Overfitting fix: dropout rate (default 0.1 if not in config) ---
        self.dropout_rate = config.get('dropoutRate', 0.1)

        # --- Class imbalance fix: weight for positive class in MSE loss ---
        # pos_weight > 1 up-weights minority class; 1.0 = no weighting
        self.pos_weight = config.get('pos_weight', 1.0)

        # Shared representation (with Dropout after each activation)
        layers = [
            nn.Linear(self.input_size, self.hidden_size),
            nn.ELU(),
            nn.Dropout(p=self.dropout_rate),   # <-- overfitting fix
        ]
        for _ in range(self.num_layers - 1):
            layers.append(nn.Linear(self.hidden_size, self.hidden_size))
            layers.append(nn.ELU())
            layers.append(nn.Dropout(p=self.dropout_rate))  # <-- overfitting fix
        self.shared_layers = nn.Sequential(*layers)

        # Head networks (one per dosage bin)
        self.head_layers = nn.ModuleList()
        for _ in range(self.num_heads):
            head = [nn.Linear(self.hidden_size + 1, self.hidden_size)]
            for _ in range(self.num_layers - 1):
                head.append(nn.Linear(self.hidden_size + 1, self.hidden_size))
            head.append(nn.Linear(self.hidden_size, 1))
            self.head_layers.append(nn.ModuleList(head))

    # Forward pass
    def forward(self, x, d):
        x = self.shared_layers(x)
        d = d.reshape(-1, 1)
        res = torch.zeros(x.shape[0], self.num_heads, device=x.device)

        for i in range(self.num_heads):
            h = torch.cat((x, d), dim=1)
            h = F.elu(self.head_layers[i][0](h))
            for j in range(1, self.num_layers):
                h = torch.cat((h, d), dim=1)
                h = F.elu(self.head_layers[i][j](h))
            h = torch.sigmoid(self.head_layers[i][self.num_layers](h))
            res[:, i] = h.squeeze()

        return res

    # Training step (with logging)
    def training_step(self, batch, batch_idx):
        x, y_true, d = batch
        y_pred = self(x, d)

        # Bin dosage
        bins = torch.bucketize(
            d, torch.linspace(0, 1, self.num_heads + 1, device=d.device)
        ) - 1
        bins = torch.clamp(bins, min=0)

        # Mask loss to factual head
        y_target = y_pred.detach().clone()
        for i in range(x.shape[0]):
            y_target[i, bins[i]] = y_true[i]

        # --- Class imbalance fix: weighted MSE ---
        # Samples where the true label is 1 (loan approved / minority class)
        # receive a higher gradient weight = pos_weight.
        sample_weights = torch.where(
            y_true == 1,
            torch.full_like(y_true, self.pos_weight),
            torch.ones_like(y_true)
        )  # shape: (batch,)

        # Compute per-sample MSE across all heads, then apply weights
        per_sample_loss = ((y_pred - y_target) ** 2).mean(dim=1)  # (batch,)
        loss = (sample_weights * per_sample_loss).mean() * self.num_heads

        # Log the training loss
        self.log("train_loss", loss, prog_bar=True)
        return loss

    # Validation step
    def validation_step(self, batch, batch_idx):
        x, y_true, d = batch
        y_pred = self(x, d)

        bins = torch.bucketize(
            d, torch.linspace(0, 1, self.num_heads + 1, device=d.device)
        ) - 1
        bins = torch.clamp(bins, min=0)

        y_target = y_pred.detach().clone()
        for i in range(x.shape[0]):
            y_target[i, bins[i]] = y_true[i]

        loss = F.mse_loss(y_pred, y_target) * self.num_heads
        # Log the validation loss
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learningRate)

    def dataloader(self, dataset, shuffle=True):
        return DataLoader(
            torchDataset(dataset),
            batch_size=self.batch_size,
            shuffle=shuffle
        )

    # Train using both train and validation sets
    def trainModel(self, dataset_train, dataset_val):
        train_loader = self.dataloader(dataset_train, shuffle=True)
        val_loader = self.dataloader(dataset_val, shuffle=False)

        # --- Overfitting fix: stop early when val_loss stops improving ---
        early_stop = EarlyStopping(
            monitor="val_loss",
            patience=10,          # wait 10 validation checks with no improvement
            min_delta=1e-4,       # improvement threshold
            mode="min",
            verbose=True
        )

        trainer = pl.Trainer(
            max_steps=self.num_steps,
            accelerator="cpu",
            devices=1,
            enable_model_summary=True,
            enable_progress_bar=True,
            enable_checkpointing=False,
            log_every_n_steps=10,
            callbacks=[early_stop]  # <-- overfitting fix
        )

        trainer.fit(self, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # Predict factual outcome
    def predictObservation(self, x, d):
        outcomes = self(x, d)
        bins = torch.bucketize(
            d, torch.linspace(0, 1, self.num_heads + 1, device=d.device)
        ) - 1
        bins = torch.clamp(bins, min=0, max=self.num_heads - 1)  # fix: guard upper bound
        preds = torch.gather(outcomes, 1, bins.reshape(-1, 1))
        return preds.detach().cpu().numpy()

    # Dose–response curve
    def getDR(self, observation):
        observation = torch.tensor(observation, dtype=torch.float32)
        num_points = 65
        d_values = torch.linspace(1e-6, 1.0, num_points)
        x = observation.repeat(num_points, 1)
        dr_curve = self.predictObservation(x, d_values).squeeze()
        return d_values.numpy(), dr_curve
