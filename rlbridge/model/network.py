"""Bridge Transformer model — unified policy + value network.

Architecture:
  [HAND] [DUMMY] [CONTEXT] [action_1] ... [action_N]
  → Causal transformer
  → Policy head (bid logits or card logits) + Value head (scalar)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from rlbridge.model.config import ModelConfig


class BridgeTransformer(nn.Module):
    """Core transformer encoder for bridge sequences."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        d = config.d_model

        # Input projections for special tokens
        self.hand_proj = nn.Linear(config.hand_dim, d)
        self.dummy_proj = nn.Linear(config.dummy_dim, d)
        self.context_proj = nn.Linear(config.context_dim, d)

        # Embeddings for action tokens
        self.bid_embed = nn.Embedding(config.bid_vocab, d)
        self.card_embed = nn.Embedding(config.card_vocab, d)
        self.action_type_embed = nn.Embedding(2, d)  # 0=bid, 1=card
        self.player_embed = nn.Embedding(config.player_vocab, d)

        # Positional embedding for the full sequence
        self.pos_embed = nn.Embedding(config.max_seq_len, d)

        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=config.n_heads,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.n_layers,
        )

        self.norm = nn.LayerNorm(d)
        self.dropout = nn.Dropout(config.dropout)

        # Cache for causal masks (keyed by (seq_len, device))
        self._causal_mask_cache = {}

    def _get_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Get or create cached causal attention mask."""
        key = (seq_len, device)
        if key not in self._causal_mask_cache:
            self._causal_mask_cache[key] = (
                nn.Transformer.generate_square_subsequent_mask(seq_len, device=device)
            )
        return self._causal_mask_cache[key]

    def forward(self, batch: dict) -> torch.Tensor:
        """Forward pass through the transformer.

        Args:
            batch: dict from collate_observations()

        Returns:
            (B, d_model) — representation at the last valid position
        """
        B = batch['hand'].shape[0]
        d = self.config.d_model
        device = batch['hand'].device

        # Project special tokens: [HAND, DUMMY, CONTEXT]
        hand_tok = self.hand_proj(batch['hand']).unsqueeze(1)      # (B, 1, d)
        dummy_tok = self.dummy_proj(batch['dummy']).unsqueeze(1)    # (B, 1, d)
        ctx_tok = self.context_proj(batch['context']).unsqueeze(1)  # (B, 1, d)

        # Build action token embeddings
        action_tokens = batch['action_tokens']  # (B, L)
        action_types = batch['action_types']    # (B, L)
        player_ids = batch['player_ids']        # (B, L)
        seq_mask = batch['seq_mask']            # (B, L)
        L = action_tokens.shape[1]

        # Embed bids and cards separately, then combine based on type
        bid_emb = self.bid_embed(action_tokens.clamp(max=self.config.bid_vocab - 1))
        card_emb = self.card_embed(action_tokens.clamp(max=self.config.card_vocab - 1))

        # Select based on action type
        is_bid = (action_types == 0).unsqueeze(-1).float()  # (B, L, 1)
        action_emb = bid_emb * is_bid + card_emb * (1 - is_bid)

        # Add type and player embeddings
        action_emb = action_emb + self.action_type_embed(action_types)
        action_emb = action_emb + self.player_embed(player_ids)

        # Concatenate: [HAND, DUMMY, CONTEXT, action_1, ..., action_L]
        seq = torch.cat([hand_tok, dummy_tok, ctx_tok, action_emb], dim=1)  # (B, 3+L, d)
        total_len = 3 + L

        # Add positional embeddings
        positions = torch.arange(total_len, device=device).unsqueeze(0)  # (1, 3+L)
        seq = seq + self.pos_embed(positions)
        seq = self.dropout(seq)

        # Build attention mask: causal (cached)
        causal_mask = self._get_causal_mask(total_len, device)

        # Build padding mask: special tokens always valid, action tokens use seq_mask
        special_mask = torch.ones(B, 3, dtype=torch.bool, device=device)
        padding_mask = torch.cat([special_mask, seq_mask], dim=1)  # (B, 3+L)
        # TransformerEncoder expects True = ignore, so invert
        src_key_padding_mask = ~padding_mask

        # Run transformer
        out = self.transformer(
            seq,
            mask=causal_mask,
            src_key_padding_mask=src_key_padding_mask,
        )
        out = self.norm(out)

        # Extract representation at last valid position
        # Find last valid position for each batch element
        lengths = padding_mask.sum(dim=1).long() - 1  # (B,)
        lengths = lengths.clamp(min=0)

        # Gather last valid position
        idx = lengths.unsqueeze(1).unsqueeze(2).expand(B, 1, d)
        last_hidden = out.gather(1, idx).squeeze(1)  # (B, d)

        return last_hidden


class PolicyHead(nn.Module):
    """Dual policy head for bids and cards."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.bid_head = nn.Linear(config.d_model, config.num_bid_actions)
        self.card_head = nn.Linear(config.d_model, config.num_card_actions)

    def forward(self, hidden: torch.Tensor, legal_mask_bids: torch.Tensor,
                legal_mask_cards: torch.Tensor) -> tuple:
        """
        Args:
            hidden: (B, d_model)
            legal_mask_bids: (B, 38) bool
            legal_mask_cards: (B, 52) bool

        Returns:
            bid_logits: (B, 38) masked logits
            card_logits: (B, 52) masked logits
        """
        bid_logits = self.bid_head(hidden)
        card_logits = self.card_head(hidden)

        # Mask illegal actions with -inf
        bid_logits = bid_logits.masked_fill(~legal_mask_bids, float('-inf'))
        card_logits = card_logits.masked_fill(~legal_mask_cards, float('-inf'))

        return bid_logits, card_logits


class ValueHead(nn.Module):
    """Value head predicting IMP vs PAR."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.d_model, config.value_hidden),
            nn.ReLU(),
            nn.Linear(config.value_hidden, 1),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden: (B, d_model)

        Returns:
            value: (B, 1)
        """
        return self.net(hidden)


class BridgeModel(nn.Module):
    """Complete bridge model: transformer + policy + value heads."""

    def __init__(self, config: ModelConfig = None):
        super().__init__()
        if config is None:
            config = ModelConfig()
        self.config = config
        self.transformer = BridgeTransformer(config)
        self.policy = PolicyHead(config)
        self.value = ValueHead(config)

    def forward(self, batch: dict) -> tuple:
        """
        Returns:
            bid_logits: (B, 38)
            card_logits: (B, 52)
            value: (B, 1)
        """
        hidden = self.transformer(batch)
        bid_logits, card_logits = self.policy(
            hidden, batch['legal_mask_bids'], batch['legal_mask_cards']
        )
        value = self.value(hidden)
        return bid_logits, card_logits, value

    def get_action_and_value(self, batch: dict, temperature: float = 1.0) -> dict:
        """Sample actions and return log probs + values for self-play.

        Returns dict with:
            action: (B,) int — sampled action IDs
            log_prob: (B,) float
            value: (B,) float
            entropy: (B,) float
        """
        bid_logits, card_logits, value = self.forward(batch)

        # Determine which head to use based on legal masks
        # If any bid is legal, use bid head; else use card head
        has_bids = batch['legal_mask_bids'].any(dim=1)  # (B,)

        actions = torch.zeros(batch['hand'].shape[0], dtype=torch.long,
                              device=batch['hand'].device)
        log_probs = torch.zeros(batch['hand'].shape[0],
                                device=batch['hand'].device)
        entropies = torch.zeros(batch['hand'].shape[0],
                                device=batch['hand'].device)

        # Process bidding positions
        bid_idx = has_bids.nonzero(as_tuple=True)[0]
        if len(bid_idx) > 0:
            bl = bid_logits[bid_idx] / temperature
            dist = torch.distributions.Categorical(logits=bl)
            a = dist.sample()
            actions[bid_idx] = a
            log_probs[bid_idx] = dist.log_prob(a)
            entropies[bid_idx] = dist.entropy()

        # Process card play positions
        card_idx = (~has_bids).nonzero(as_tuple=True)[0]
        if len(card_idx) > 0:
            cl = card_logits[card_idx] / temperature
            dist = torch.distributions.Categorical(logits=cl)
            a = dist.sample()
            actions[card_idx] = a
            log_probs[card_idx] = dist.log_prob(a)
            entropies[card_idx] = dist.entropy()

        return {
            'action': actions,
            'log_prob': log_probs,
            'value': value.squeeze(-1),
            'entropy': entropies,
        }

    def evaluate_actions(self, batch: dict, actions: torch.Tensor,
                         is_bid: torch.Tensor) -> dict:
        """Evaluate given actions (for PPO update).

        Args:
            batch: collated batch
            actions: (B,) int — action IDs
            is_bid: (B,) bool — True if action is a bid

        Returns:
            dict with log_prob, value, entropy
        """
        bid_logits, card_logits, value = self.forward(batch)

        log_probs = torch.zeros_like(value.squeeze(-1))
        entropies = torch.zeros_like(value.squeeze(-1))

        bid_idx = is_bid.nonzero(as_tuple=True)[0]
        if len(bid_idx) > 0:
            dist = torch.distributions.Categorical(logits=bid_logits[bid_idx])
            log_probs[bid_idx] = dist.log_prob(actions[bid_idx])
            entropies[bid_idx] = dist.entropy()

        card_idx = (~is_bid).nonzero(as_tuple=True)[0]
        if len(card_idx) > 0:
            dist = torch.distributions.Categorical(logits=card_logits[card_idx])
            log_probs[card_idx] = dist.log_prob(actions[card_idx])
            entropies[card_idx] = dist.entropy()

        return {
            'log_prob': log_probs,
            'value': value.squeeze(-1),
            'entropy': entropies,
        }
