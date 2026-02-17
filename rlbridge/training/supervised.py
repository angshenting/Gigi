"""Optional: supervised pre-training from BEN bidding data.

This module provides utilities to bootstrap the model from existing
BEN bidding data before self-play training.
"""

import os
import logging

import torch
import torch.nn.functional as F
import numpy as np

from rlbridge.model.config import ModelConfig
from rlbridge.model.network import BridgeModel

logger = logging.getLogger(__name__)


class SupervisedTrainer:
    """Pre-train the bridge model on supervised bidding data."""

    def __init__(self, model: BridgeModel, lr: float = 1e-4,
                 device: str = 'cpu'):
        self.model = model
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.model.to(device)

    def train_epoch(self, dataloader) -> dict:
        """Train one epoch on supervised data.

        Args:
            dataloader: yields batches of (batch_dict, target_actions, is_bid)

        Returns:
            dict of metrics
        """
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for batch, targets, is_bid in dataloader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            targets = targets.to(self.device)
            is_bid = is_bid.to(self.device)

            bid_logits, card_logits, _ = self.model(batch)

            # Compute cross-entropy loss
            bid_idx = is_bid.nonzero(as_tuple=True)[0]
            card_idx = (~is_bid).nonzero(as_tuple=True)[0]

            loss = torch.tensor(0.0, device=self.device)
            if len(bid_idx) > 0:
                loss = loss + F.cross_entropy(
                    bid_logits[bid_idx], targets[bid_idx]
                )
            if len(card_idx) > 0:
                loss = loss + F.cross_entropy(
                    card_logits[card_idx], targets[card_idx]
                )

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            # Accuracy
            with torch.no_grad():
                if len(bid_idx) > 0:
                    pred_bids = bid_logits[bid_idx].argmax(dim=-1)
                    total_correct += (pred_bids == targets[bid_idx]).sum().item()
                if len(card_idx) > 0:
                    pred_cards = card_logits[card_idx].argmax(dim=-1)
                    total_correct += (pred_cards == targets[card_idx]).sum().item()

            total_loss += loss.item() * len(targets)
            total_samples += len(targets)

        return {
            'loss': total_loss / max(total_samples, 1),
            'accuracy': total_correct / max(total_samples, 1),
        }
