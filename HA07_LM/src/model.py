# -*- coding: utf-8 -*-

"""
Student Name: Liu Shanru
Student ID: 21190664
Student Email: sliufo@connect.ust.hk
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import FloatTensor,LongTensor, BoolTensor
from typing import Optional, Iterable


class SelfAttention(nn.Module):
    r"""A self-attention layer.

    This module takes inputs :math:`X\in\mathbb R^{N\times L\times D_e}`, and projects them into
    queries :math:`Q\in\mathbb R^{N\times L\times D_k}`, keys :math:`K\in\mathbb R^{N\times L\times D_k}`,
    and values :math:`V\in\mathbb R^{N\times L\times D_v}`, where
    :math:`N` is the batch size, :math:`L` is the padded sequence length.
    Accordingly the layer outputs in shape :math:`(N, L, D_v)`.

    Args:
        emb_dim: The dimension of embeddings, i.e. D_e.
        key_dim: The dimension of keys, i.e. D_k.
        val_dim: The dimension of values, i.e. D_v.
    """

    def __init__(self, emb_dim: int, key_dim: int, val_dim: int) -> None:
        super().__init__()

        self.emb_dim:int = emb_dim
        r"""
        The dimension of embedings, i.e. :math:`D_e`.
        """
        self.key_dim:int = key_dim
        r"""
        The dimension of keys, i.e. :math:`D_k`.
        r"""
        self.val_dim:int = val_dim
        r"""
        The dimension of values, i.e. :math:`D_v`.
        """

        self.proj_q:nn.Parameter = nn.Parameter(torch.empty(self.emb_dim, self.key_dim))
        r"""
        The projection matrix for queries :math:`M_q\in\mathbb R^{D_e, D_v}`.
        """
        self.proj_k:nn.Parameter = nn.Parameter(torch.empty(self.emb_dim, self.key_dim))
        r"""
        The projection matrix for keys :math:`M_k\in\mathbb R^{D_e, D_k}`.
        """
        self.proj_v:nn.Parameter = nn.Parameter(torch.empty(self.emb_dim, self.val_dim))
        r"""
        The projection matrix for values :math:`M_v\in\mathbb R^{D_e, D_v}`.
        """

        nn.init.xavier_uniform_(self.proj_q)
        nn.init.xavier_uniform_(self.proj_k)
        nn.init.xavier_uniform_(self.proj_v)

    def forward(self, x: FloatTensor, attn_mask: Optional[BoolTensor] = None) -> FloatTensor:
        r"""Compute self-attention output.

        Todo:
            #. Compute queries :math:`Q`, keys :math:`K`,
               and values :math:`V` with input embeddings :math:`x`.
            #. Compute the (masked) self-attention map.
            #. Compute the last output.

        Args:
            x: The input of shape :math:`(N, L, D_e)`.
            attn_mask: The optional attention mask of shape :math:`(N, L, L)`.

        Returns:
            The self-attention output of shape :math:`(N, L, D_v)`.
        """
        # Validate arguments.
        assert (
            x.dim() == 3 and x.size(-1) == self.emb_dim
        ), "The input shape should be of (N, L, D_e)!"

        N, L, D_e = x.size()
        if attn_mask is not None:
            assert (
                attn_mask.size() == (N, L, L)
            ), "The attention mask must be of shape (N, L, L)!"
            assert (
                attn_mask.dtype == torch.bool
            ), "The attention mask must be a boolean tensor!"
        else:
            attn_mask = torch.ones(N, L, L, dtype=torch.bool, device=x.device)

        # Compute self-attention output.

        ######################################### YOUR CODE HERE ##################################
        # 1. Q, K, V
        Q = torch.matmul(x, self.proj_q)
        K = torch.matmul(x, self.proj_k)
        V = torch.matmul(x, self.proj_v)

        # 2. Attention score
        attn_score = torch.matmul(Q, K.transpose(-2, -1)) / (self.key_dim ** 0.5)

        # 3. Apply mask
        attn_score = attn_score.masked_fill(~attn_mask, float('-inf'))

        # 4. attention map
        attn_weights = torch.softmax(attn_score, dim=-1) # (N, L, L)
        out = torch.matmul(attn_weights, V) #(N, L, D_v)

        ######################################### END OF YOUR CODE ################################
        return out


class SFLM(nn.Module):
    r"""A Small formal language model.

    This module takes sequences of indices of tokens in shape :math:`(N, L)`, where :math:`N` is the batch size,
    :math:`L` is the padded length of sequences, and outputs logits of shape :math:`(N, L, V)` for predicting next
    token at each position, where :math:`L` is the size of the vocabulary.

    .. _SFLM_structure:
    .. figure:: /../SFLM_architecture.png
        :width: 50%

        The model structure.

    Args:
        vocab_size: The size of the vocabulary, i.e. :math:`V`.
        emb_dim: The dimension of embeddings.
        block_size: The maximum length of input sequences, i.e. maximum value of :math:`L`.
    """

    def __init__(self, vocab_size: int, emb_dim: int, block_size: int) -> None:
        super().__init__()
        self.vocab_size:int = vocab_size
        r"""The size of the vocabulary, i.e. V."""
        self.emb_dim:int = emb_dim
        r"""The dimension of embeddings."""
        self.block_size:int = block_size
        r"""The maximum length of input sequences."""

        self.tok_embedding:nn.Embedding = nn.Embedding(
            num_embeddings=self.vocab_size, embedding_dim=self.emb_dim
        )
        r"""The embedding layer for translating token indices into embeddings."""
        self.pos_embedding:nn.Embedding = nn.Embedding(
            num_embeddings=self.block_size, embedding_dim=self.emb_dim
        )
        r"""The embedding layer for encoding the position of the corresponding token."""
        self.self_attention: SelfAttention = SelfAttention(
            emb_dim=self.emb_dim, key_dim=self.emb_dim, val_dim=self.emb_dim
        )
        r"""The self-attention layer. Here, the dimension of keys and values are set to emb_dim."""
        self.layer_norm_1: nn.LayerNorm = nn.LayerNorm(self.emb_dim)
        r"""The layer normalization for self-attention layer."""
        self.fnn: nn.Sequential = nn.Sequential(
            nn.Linear(self.emb_dim, 4 * self.emb_dim),
            nn.ReLU(),
            nn.Linear(4 * self.emb_dim, self.emb_dim),
        )
        r"""The FNN layer."""
        self.layer_norm_2:nn.LayerNorm = nn.LayerNorm(self.emb_dim)
        r"""The layer normalization for FNN layer."""
        self.head:nn.Linear = nn.Linear(self.emb_dim, self.vocab_size)
        r"""The linear layer project embeddings into logits for predicting the next token at each position."""

    def forward(self, idx: LongTensor) -> FloatTensor:
        r"""Compute logits of the next token.

        Denote input as :math:`S`, and output as :math:`Z`,
        :math:`Z_{i,j,k}` represents the logit of :math:`S_{i,j+1}`
        to be the :math:`k`-th token given :math:`S_{i,1:j}`.


        .. _SFLM_forward:
        .. figure:: /../SFLM_forward.png

            The forward process.

        Todo:
            Complete the forward function of this SFLM model.
            Refer to the architecture illustrated :ref:`here<SFLM_structure>`.

            The model must be **CAUSAL** in the aspect of the sequence order as
            shown :ref:`above<SFLM_forward>`,
            i.e. at each position, the model cannot access any token after it.
            You can achieve this requirement by applying a proper attention mask.

        Args:
            idx: The input of shape :math:`(N, L)`.

        Returns:
            The language model output of shape :math:`(N, L, V)` for predicting the next token.
        """
        # Validate arguments.
        assert idx.dim() == 2, "Indices must be of shape (N, L)!"
        assert idx.dtype == torch.long, "Indices must be of type Long!"
        N, L = idx.size()
        assert L <= self.block_size, "Sequences are too long!"

        # Compute the SFLM output.
        
        ######################################### YOUR CODE HERE ##################################
        # token embedding (N, L) -> (N, L, De)
        tok_emb = self.tok_embedding(idx)

        # position embedding
        pos = torch.arange(L, device=idx.device)
        pos_emb = self.pos_embedding(pos) #(L, De)

        # combine token and position embedding
        x = tok_emb + pos_emb #(N, L, De)

        # Build causal mask so that the model can only atten to the earlier ones
        causal_mask = torch.tril(torch.ones(L, L, dtype=torch.bool, device=idx.device))
        causal_mask = causal_mask.unsqueeze(0).expand(N, -1, -1) #(N, L, L)

        # self-attention
        attn_out = self.self_attention(x, attn_mask=causal_mask) #(N, L, De)
        x = x + attn_out
        x = self.layer_norm_1(x)

        # feed-forward
        fnn_out = self.fnn(x)
        x = x + fnn_out
        x = self.layer_norm_2(x)

        # output logits
        logit = self.head(x) #(N, L, V)

        ######################################### END OF YOUR CODE ################################
        return logit

    @torch.no_grad()
    def generate(
        self, cond_idx: LongTensor, steps: int, temperature: Optional[float] = 1.0
    ) -> LongTensor:
        r"""Conditional sample from this language model.

        .. _SFLM_generation:
        .. figure:: /../SFLM_generate.png

            Given a single BOS as the condition, a sample "abcc" is generated.

        Args:
            cond_idx: The input of shape :math:`(N, L)`.
                It represents the indices of the first :math:`L` token given as condition.
            steps: The steps for generation.
            temperature: The temperature for sampling, default to 1.0.
                For greedy strategy, just give 0.0.

        Returns:
            The sampled indices of shape :math:`(N, L + \textit{steps})`. When :math:`L + \textit{steps}` is greater than
            block_size, the generation would always depends the last block_size tokens in a moving window.
        """
        assert cond_idx.dim() == 2, "Condition indices must be of shape (N, L)!"
        assert cond_idx.dtype == torch.long, "Condition indices must be of type Long!"
        assert temperature >= 0, "Temperature cannot be less than zero by definition!"
        N, L = cond_idx.size()

        idx = cond_idx.clone()
        for _ in range(steps):
            logit = self(idx[:, -self.block_size :])
            next_token_logit = logit[:, -1, :]
            if temperature > 0:
                next_token_prob = torch.softmax(next_token_logit / temperature, -1)
                next_token_id = torch.multinomial(next_token_prob, num_samples=1)
            else:
                next_token_id = torch.argmax(next_token_logit, dim=-1, keepdim=True)
            idx = torch.cat([idx, next_token_id], -1)
        return idx




        # TODO1: Compute token and position embeddings
        tok_emb = self.tok_embedding(idx)  # (N, L, D_e)
        pos_ids = torch.arange(L, device=idx.device).unsqueeze(0).expand(N, -1)  # (N, L)
        pos_emb = self.pos_embedding(pos_ids)  # (N, L, D_e)
        x = tok_emb + pos_emb  # (N, L, D_e)

        # TODO2: Apply the self-attention layer with proper attention mask
        causal_mask = torch.triu(torch.ones(L, L), diagonal=1).bool().to(x.device)  # (L, L)
        attn_mask = ~causal_mask.unsqueeze(0).expand(N, -1, -1)  # (N, L, L)
        x2 = self.self_attention(x, attn_mask=attn_mask)  # (N, L, D_e)
        x = self.layer_norm_1(x + x2)  # (N, L, D_e)

        # TODO3: Apply the FNN layer
        x3 = self.fnn(x)  # (N, L, D_e)
        x = self.layer_norm_2(x + x3)  # (N, L, D_e)

        # TODO4: Compute the final output logits
        logit = self.head(x)  # (N, L, V)   