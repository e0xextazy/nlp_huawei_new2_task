from typing import Dict

import torch
from numpy import asarray
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from tqdm.notebook import tqdm

from model import RecurrentClassifier


class Trainer:
    def __init__(self, config: Dict):
        """
        Fits end evaluates given model with Adam optimizer.
         Hyperparameters are specified in `config`
        Possible keys are:
            - n_epochs: number of epochs to train
            - lr: optimizer learning rate
            - weight_decay: l2 regularization weight
            - device: on which device to perform training ("cpu" or "cuda")
            - verbose: whether to print anything during training
        :param config: configuration for `Trainer`
        """
        self.config = config
        self.n_epochs = config["n_epochs"]
        self.setup_opt_fn = lambda model: Adam(
            model.parameters(), config["lr"], weight_decay=config["weight_decay"]
        )
        self.model = None
        self.opt = None
        self.history = None
        self.loss_fn = CrossEntropyLoss()
        self.device = config["device"]
        self.verbose = config.get("verbose", True)

    def fit(self, model, train_loader, val_loader):
        """
        Fits model on training data, each epoch evaluates on validation data
        :param model: PyTorch model for toxic comments classification (for example, `RecurrentClassifier`)
        :param train_loader: DataLoader for training data
        :param val_loader: DataLoader for validation data
        :return:
        """
        self.model = model.to(self.device)
        self.opt = self.setup_opt_fn(self.model)
        self.history = {"train_loss": [], "val_loss": [], "val_acc": []}
        for epoch in range(self.n_epochs):
            print(f"Epoch {epoch + 1}/{self.n_epochs}")
            train_info = self._train_epoch(train_loader)
            val_info = self._val_epoch(val_loader)
            self.history["train_loss"].extend(train_info["train_loss"])
            self.history["val_loss"].append(val_info["loss"])
            self.history["val_acc"].append(val_info["acc"])
        return self.model.eval()

    def _train_epoch(self, train_loader):
        self.model.train()
        losses = []
        if self.verbose:
            train_loader = tqdm(train_loader)
        for batch in train_loader:
            self.model.zero_grad()
            texts, labels = batch
            logits = self.model.forward(texts.to(self.device))
            loss = self.loss_fn(logits, labels.to(self.device))
            loss.backward()
            self.opt.step()
            loss_val = loss.item()
            if self.verbose:
                train_loader.set_description(f"Loss={loss_val:.3}")
            losses.append(loss_val)
        return {"train_loss": losses}

    def _val_epoch(self, val_loader):
        self.model.eval()
        all_logits = []
        all_labels = []
        if self.verbose:
            val_loader = tqdm(val_loader)
        with torch.no_grad():
            for batch in val_loader:
                texts, labels = batch
                logits = self.model.forward(texts.to(self.device))
                all_logits.append(logits)
                all_labels.append(labels)
        all_labels = torch.cat(all_labels).to(self.device)
        all_logits = torch.cat(all_logits)
        loss = CrossEntropyLoss()(all_logits, all_labels).item()
        acc = (all_logits.argmax(1) == all_labels).float().mean().item()
        if self.verbose:
            val_loader.set_description(f"Loss={loss:.3}; Acc:{acc:.3}")
        return {"acc": acc, "loss": loss}

    def predict(self, test_loader):
        if self.model is None:
            raise RuntimeError("You should train the model first")
        self.model.eval()
        predictions = []
        with torch.no_grad():
            for batch in test_loader:
                texts, labels = batch
                logits = self.model.forward(texts.to(self.device))
                predictions.extend(logits.argmax(1).tolist())
        return asarray(predictions)

    def save(self, path: str):
        if self.model is None:
            raise RuntimeError("You should train the model first")
        checkpoint = {
            "config": self.model.config,
            "trainer_config": self.config,
            "vocab": self.model.vocab,
            "emb_matrix": self.model.emb_matrix,
            "state_dict": self.model.state_dict(),
        }
        torch.save(checkpoint, path)

    @classmethod
    def load(cls, path: str):
        ckpt = torch.load(path)
        keys = ["config", "trainer_config",
                "vocab", "emb_matrix", "state_dict"]
        for key in keys:
            if key not in ckpt:
                raise RuntimeError(f"Missing key {key} in checkpoint")
        new_model = RecurrentClassifier(
            ckpt["config"], ckpt["vocab"], ckpt["emb_matrix"]
        )
        new_model.load_state_dict(ckpt["state_dict"])
        new_trainer = cls(ckpt["trainer_config"])
        new_trainer.model = new_model
        new_trainer.model.to(new_trainer.device)
        return new_trainer
