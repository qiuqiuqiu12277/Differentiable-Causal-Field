"""
MetaCausalField: Differentiable Causal Field for Multimodal Causal Reasoning

This module implements the core components of MetaCausalField, a unified framework 
that replaces the three-stage pipeline of MLLM-CD (factor discovery → structure learning 
→ counterfactual refinement) with an end-to-end differentiable causal field.

Key innovations:
1. Continuous Causal Field: Represents causal factors as continuous spatial distributions 
   rather than discrete variables
2. Directional Influence Function: Models causal relationships as continuous influence 
   propagation instead of static graph edges
3. Field-based Intervention: Enables direct manipulation of intermediate representations 
   for counterfactual reasoning
4. Dynamic Evolution: Captures temporal propagation of causal effects through iterative 
   field updates

Reference: "MetaCausalField: Unified Multimodal Causal Modeling via Differentiable 
Causal Fields" (NeurIPS 2025 submission)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict, List, Union
from dataclasses import dataclass
import math
import re
from collections import Counter


@dataclass
class CausalFieldConfig:
    """Configuration for Causal Field construction and propagation.
    
    Attributes:
        spatial_size: Spatial dimensions of the causal field (H, W)
        feature_dim: Dimension of feature vectors at each spatial position
        num_heads: Number of attention heads for influence function
        num_propagation_steps: Number of iterative propagation steps
        intervention_radius: Spatial radius for local interventions
        gaussian_sigma: Standard deviation for Gaussian interpolation weights
        use_temporal_dynamics: Whether to model temporal evolution
        temporal_horizon: Number of future steps to predict for counterfactuals
        lambda_consistency: Weight for cross-environment invariance loss
        lambda_counterfactual: Weight for counterfactual rollout supervision
        lambda_sparsity: Weight for sparsity regularization
        lambda_smoothness: Weight for spatial smoothness regularization
        influence_top_k: Optional top-k outgoing influences to retain per location
        dropout: Dropout rate for regularization
    """
    spatial_size: Tuple[int, int] = (14, 14)  # Default ViT patch grid
    feature_dim: int = 512
    num_heads: int = 8
    num_propagation_steps: int = 3
    intervention_radius: float = 2.0
    gaussian_sigma: float = 1.0
    use_temporal_dynamics: bool = True
    temporal_horizon: int = 5
    lambda_consistency: float = 0.3
    lambda_counterfactual: float = 0.5
    lambda_sparsity: float = 0.01
    lambda_smoothness: float = 0.001
    influence_top_k: Optional[int] = None
    dropout: float = 0.1
    vocab_size: int = 4096
    max_text_length: int = 96
    num_factor_classes: int = 3
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    unk_token_id: int = 3


class SimpleTokenizer:
    """Small word-level tokenizer used by the local training and inference scripts.

    It keeps the project self-contained while still providing a real language
    objective. The vocabulary is saved in checkpoints and can be rebuilt from
    the dataset when training from scratch.
    """

    PAD = "<pad>"
    BOS = "<bos>"
    EOS = "<eos>"
    UNK = "<unk>"

    def __init__(self, vocab: Optional[Dict[str, int]] = None, max_length: int = 96):
        self.max_length = max_length
        self.token_to_id = vocab or {
            self.PAD: 0,
            self.BOS: 1,
            self.EOS: 2,
            self.UNK: 3,
        }
        self.id_to_token = {idx: tok for tok, idx in self.token_to_id.items()}

    @staticmethod
    def tokenize(text: str) -> List[str]:
        return re.findall(r"[A-Za-z0-9]+|[^\w\s]", str(text).lower())

    @classmethod
    def build(cls, texts: List[str], vocab_size: int = 4096, min_freq: int = 1, max_length: int = 96):
        counter = Counter()
        for text in texts:
            counter.update(cls.tokenize(text))

        vocab = {
            cls.PAD: 0,
            cls.BOS: 1,
            cls.EOS: 2,
            cls.UNK: 3,
        }
        for token, freq in counter.most_common(max(0, vocab_size - len(vocab))):
            if freq >= min_freq and token not in vocab:
                vocab[token] = len(vocab)
        return cls(vocab=vocab, max_length=max_length)

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    def encode(self, text: str, max_length: Optional[int] = None) -> List[int]:
        length = max_length or self.max_length
        ids = [self.token_to_id[self.BOS]]
        ids.extend(self.token_to_id.get(tok, self.token_to_id[self.UNK]) for tok in self.tokenize(text))
        ids.append(self.token_to_id[self.EOS])
        ids = ids[:length]
        if ids[-1] != self.token_to_id[self.EOS] and length > 1:
            ids[-1] = self.token_to_id[self.EOS]
        ids.extend([self.token_to_id[self.PAD]] * (length - len(ids)))
        return ids

    def decode(self, ids: Union[List[int], torch.Tensor]) -> str:
        if isinstance(ids, torch.Tensor):
            ids = ids.detach().cpu().tolist()
        tokens = []
        for idx in ids:
            token = self.id_to_token.get(int(idx), self.UNK)
            if token == self.EOS:
                break
            if token not in {self.PAD, self.BOS}:
                tokens.append(token)
        text = " ".join(tokens)
        text = re.sub(r"\s+([.,;:!?])", r"\1", text)
        return text

    def state_dict(self) -> Dict[str, Union[Dict[str, int], int]]:
        return {"token_to_id": self.token_to_id, "max_length": self.max_length}

    @classmethod
    def from_state_dict(cls, state: Dict[str, Union[Dict[str, int], int]]):
        return cls(vocab=state["token_to_id"], max_length=int(state.get("max_length", 96)))


class VisualEncoder(nn.Module):
    """ResNet patch/video encoder that returns visual tokens instead of one pooled vector."""

    def __init__(
        self,
        feature_dim: int = 512,
        pretrained: bool = False,
        train_backbone: bool = True,
        max_temporal_frames: int = 64,
        temporal_pooling: str = "attention",
    ):
        super().__init__()
        import torchvision.models as models

        weights = None
        if pretrained:
            try:
                weights = models.ResNet18_Weights.DEFAULT
            except AttributeError:
                weights = "DEFAULT"
        resnet = models.resnet18(weights=weights)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.proj = nn.Sequential(
            nn.Conv2d(512, feature_dim, kernel_size=1),
            nn.GroupNorm(1, feature_dim),
            nn.GELU(),
        )
        self.feature_dim = feature_dim
        self.temporal_pooling = temporal_pooling
        self.temporal_pos_embed = nn.Embedding(max_temporal_frames, feature_dim)
        self.temporal_query = nn.Parameter(torch.randn(feature_dim) * 0.02)
        nhead = next((h for h in (8, 4, 2, 1) if feature_dim % h == 0), 1)
        temporal_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=nhead,
            dim_feedforward=feature_dim * 2,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )
        self.temporal_encoder = nn.TransformerEncoder(temporal_layer, num_layers=1)
        self.temporal_norm = nn.LayerNorm(feature_dim)

        if not train_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def _encode_frames(self, images: torch.Tensor) -> torch.Tensor:
        features = self.backbone(images)
        features = self.proj(features)
        B, C, H, W = features.shape
        return features.flatten(2).transpose(1, 2).contiguous()

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.dim() == 5:
            B, T, C, H, W = images.shape
            frame_tokens = self._encode_frames(images.reshape(B * T, C, H, W))
            _, N, D = frame_tokens.shape
            frame_tokens = frame_tokens.reshape(B, T, N, D)
            positions = torch.arange(T, device=images.device).clamp(max=self.temporal_pos_embed.num_embeddings - 1)
            frame_tokens = frame_tokens + self.temporal_pos_embed(positions).view(1, T, 1, D)
            temporal_tokens = frame_tokens.permute(0, 2, 1, 3).reshape(B * N, T, D)
            temporal_tokens = self.temporal_encoder(temporal_tokens)
            if self.temporal_pooling == "mean":
                temporal_tokens = temporal_tokens.mean(dim=1)
            else:
                scores = torch.matmul(temporal_tokens, self.temporal_query.to(temporal_tokens.dtype))
                weights = torch.softmax(scores / math.sqrt(D), dim=1).unsqueeze(-1)
                temporal_tokens = (temporal_tokens * weights).sum(dim=1)
            temporal_tokens = self.temporal_norm(temporal_tokens)
            return temporal_tokens.reshape(B, N, D).contiguous()
        return self._encode_frames(images)


class TextEncoder(nn.Module):
    """Transformer text encoder used to query and condition the causal field."""

    def __init__(self, vocab_size: int, feature_dim: int, max_length: int, dropout: float = 0.1):
        super().__init__()
        nhead = next((h for h in (8, 4, 2, 1) if feature_dim % h == 0), 1)
        self.token_embed = nn.Embedding(vocab_size, feature_dim, padding_idx=0)
        self.pos_embed = nn.Embedding(max_length, feature_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=nhead,
            dim_feedforward=feature_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.norm = nn.LayerNorm(feature_dim)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, L = input_ids.shape
        positions = torch.arange(L, device=input_ids.device).unsqueeze(0).expand(B, L)
        hidden = self.token_embed(input_ids) + self.pos_embed(positions)
        padding_mask = input_ids.eq(0)
        hidden = self.encoder(hidden, src_key_padding_mask=padding_mask)
        return self.norm(hidden)


class TextCausalDecoder(nn.Module):
    """Transformer decoder that generates text from the causal field memory."""

    def __init__(self, vocab_size: int, feature_dim: int, max_length: int, dropout: float = 0.1):
        super().__init__()
        nhead = next((h for h in (8, 4, 2, 1) if feature_dim % h == 0), 1)
        self.token_embed = nn.Embedding(vocab_size, feature_dim, padding_idx=0)
        self.pos_embed = nn.Embedding(max_length, feature_dim)
        layer = nn.TransformerDecoderLayer(
            d_model=feature_dim,
            nhead=nhead,
            dim_feedforward=feature_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=2)
        self.norm = nn.LayerNorm(feature_dim)
        self.lm_head = nn.Linear(feature_dim, vocab_size)

    def forward(self, decoder_input_ids: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        B, L = decoder_input_ids.shape
        positions = torch.arange(L, device=decoder_input_ids.device).unsqueeze(0).expand(B, L)
        hidden = self.token_embed(decoder_input_ids) + self.pos_embed(positions)
        causal_mask = torch.triu(
            torch.ones(L, L, device=decoder_input_ids.device, dtype=torch.bool),
            diagonal=1,
        )
        padding_mask = decoder_input_ids.eq(0)
        hidden = self.decoder(
            tgt=hidden,
            memory=memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=padding_mask,
        )
        return self.lm_head(self.norm(hidden))


class GaussianInterpolation(nn.Module):
    """Continuous field construction via Gaussian interpolation.
    
    Maps discrete patch features to continuous spatial field using
    Gaussian-weighted interpolation (Equation in Section 3.2):
    
        F(p, 0) = Σ_i w_i(p) · z_i
        where w_i(p) = exp(-||p - p_i||² / σ²) / Σ_j exp(-||p - p_j||² / σ²)
    
    This enables:
    - Continuous spatial representation from discrete patches
    - Querying field values at arbitrary positions
    - Smooth spatial transitions between features
    """
    
    def __init__(self, sigma: float = 1.0):
        super().__init__()
        self.sigma = sigma
        self.sigma_sq = sigma ** 2
    
    def forward(self, 
                features: torch.Tensor, 
                patch_positions: torch.Tensor,
                query_positions: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            features: [B, N, C] discrete patch features (N patches, C dimensions)
            patch_positions: [B, N, 2] spatial positions of patches (x, y normalized to [0, 1])
            query_positions: [B, H*W, 2] query positions (default: regular grid)
            
        Returns:
            field: [B, H, W, C] or [B, Q, C] interpolated field values
        """
        B, N, C = features.shape
        device = features.device
        
        if query_positions is None:
            # Create regular grid query positions
            H, W = int(math.sqrt(N)), int(math.sqrt(N))
            y_grid = torch.linspace(0, 1, H, device=device)
            x_grid = torch.linspace(0, 1, W, device=device)
            yy, xx = torch.meshgrid(y_grid, x_grid, indexing='ij')
            query_positions = torch.stack([xx, yy], dim=-1).reshape(1, H * W, 2).expand(B, -1, -1)
        else:
            H = W = int(math.sqrt(query_positions.shape[1]))
        
        Q = query_positions.shape[1]
        
        # Compute pairwise distances: [B, Q, N]
        # ||p - p_i||² = ||p||² + ||p_i||² - 2·p·p_i
        query_expanded = query_positions.unsqueeze(2)  # [B, Q, 1, 2]
        patch_expanded = patch_positions.unsqueeze(1)   # [B, 1, N, 2]
        
        dist_sq = torch.sum((query_expanded - patch_expanded) ** 2, dim=-1)  # [B, Q, N]
        
        # Gaussian weights
        weights = torch.exp(-dist_sq / (2 * self.sigma_sq))  # [B, Q, N]
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)  # Normalize
        
        # Interpolate: [B, Q, N] @ [B, N, C] -> [B, Q, C]
        field_values = torch.bmm(weights, features)  # [B, Q, C]
        
        if query_positions.shape[1] == H * W:
            field_values = field_values.reshape(B, H, W, C)
        
        return field_values


class DirectionalInfluenceFunction(nn.Module):
    """Directional causal influence modeling between spatial positions.
    
    Implements the influence function G(p_i → p_j) from Section 3.3:
    
        G(p_i, p_j) = MLP([F(p_i), F(p_j)])
    
    Unlike attention which measures correlation, this function explicitly models
    directed causal influence with the following properties:
    - Asymmetry: G(p_i → p_j) ≠ G(p_j → p_i) in general
    - Locality: Influence decays with spatial distance
    - Sparsity: Most positions have minimal influence on each other
    
    The learned influence matrix G ∈ R^{N×N} serves as the continuous analog
    of discrete causal graph edges.
    """
    
    def __init__(
        self,
        feature_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        top_k: Optional[int] = None,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads
        self.top_k = top_k
        
        assert self.head_dim * num_heads == feature_dim, "feature_dim must be divisible by num_heads"
        
        # MLP for influence computation with multi-head structure
        self.query_proj = nn.Linear(feature_dim, feature_dim)
        self.key_proj = nn.Linear(feature_dim, feature_dim)
        self.value_proj = nn.Linear(feature_dim, feature_dim)
        
        # Causal direction modeling (asymmetric)
        self.direction_bias = nn.Parameter(torch.zeros(1, num_heads, 1, 1))
        self.directional_bias_proj = nn.Linear(3, num_heads, bias=False)
        
        # Output projection
        self.out_proj = nn.Linear(feature_dim, feature_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Sparsity-promoting mask (learnable)
        self.sparsity_gate = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, 1),
            nn.Sigmoid()
        )
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.query_proj.weight)
        nn.init.xavier_uniform_(self.key_proj.weight)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.directional_bias_proj.weight)
    
    @staticmethod
    def _default_positions(
        batch_size: int,
        num_positions: int,
        device: torch.device,
        dtype: torch.dtype,
        height: Optional[int] = None,
        width: Optional[int] = None,
    ) -> torch.Tensor:
        if height is None or width is None:
            height = max(1, int(math.sqrt(num_positions)))
            width = int(math.ceil(num_positions / height))
        y_grid = torch.linspace(0, 1, height, device=device, dtype=dtype)
        x_grid = torch.linspace(0, 1, width, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(y_grid, x_grid, indexing='ij')
        coords = torch.stack([xx, yy], dim=-1).reshape(1, height * width, 2)[:, :num_positions]
        return coords.expand(batch_size, -1, -1)

    def forward(
        self,
        field: torch.Tensor,
        spatial_positions: Optional[torch.Tensor] = None,
        return_components: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Args:
            field: [B, H, W, C] or [B, N, C] causal field
            spatial_positions: [B, N, 2] normalized coordinates for directional bias
            
        Returns:
            influence_matrix: [B, N, N] directed influence weights G(p_i → p_j)
            output_field: [B, N, C] updated field after influence aggregation
        """
        if field.dim() == 4:
            B, H, W, C = field.shape
            N = H * W
            field = field.reshape(B, N, C)
        else:
            B, N, C = field.shape
            H = max(1, int(math.sqrt(N)))
            W = int(math.ceil(N / H))

        if spatial_positions is None:
            spatial_positions = self._default_positions(B, N, field.device, field.dtype, H, W)
        elif spatial_positions.dim() == 2:
            spatial_positions = spatial_positions.unsqueeze(0).expand(B, -1, -1)
        spatial_positions = spatial_positions.to(device=field.device, dtype=field.dtype)
        
        # Compute queries, keys, values
        Q = self.query_proj(field).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, N, D]
        K = self.key_proj(field).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)    # [B, H, N, D]
        V = self.value_proj(field).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, N, D]
        
        # Compute attention scores (influence strength)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)  # [B, H, N, N]
        
        # Add a learned bias from signed relative coordinates. The scalar
        # head bias is kept for compatibility, while the relative-coordinate
        # branch makes the bias genuinely directional.
        delta = spatial_positions.unsqueeze(2) - spatial_positions.unsqueeze(1)  # [B, N, N, 2]
        distance = torch.norm(delta, dim=-1, keepdim=True)
        directional_features = torch.cat([delta, distance], dim=-1)
        directional_bias = self.directional_bias_proj(directional_features).permute(0, 3, 1, 2)
        scores = scores + self.direction_bias + directional_bias
        
        # Apply sparsity gate based on feature concatenation
        field_expanded_i = field.unsqueeze(2).expand(-1, -1, N, -1)  # [B, N, N, C]
        field_expanded_j = field.unsqueeze(1).expand(-1, N, -1, -1)  # [B, N, N, C]
        field_pair = torch.cat([field_expanded_i, field_expanded_j], dim=-1)  # [B, N, N, 2C]
        
        sparsity_weight = self.sparsity_gate(field_pair).squeeze(-1)  # [B, N, N]
        
        # Combine attention with sparsity
        influence = torch.softmax(scores, dim=-1)  # [B, H, N, N]
        
        # Average over heads and apply sparsity
        influence_matrix = influence.mean(dim=1) * sparsity_weight  # [B, N, N]
        
        # Renormalize after sparsity
        influence_matrix = influence_matrix / (influence_matrix.sum(dim=-1, keepdim=True) + 1e-8)

        if self.top_k is not None and 0 < self.top_k < N:
            values, indices = torch.topk(influence_matrix, k=int(self.top_k), dim=-1)
            sparse = torch.zeros_like(influence_matrix).scatter_(-1, indices, values)
            influence_matrix = sparse / (sparse.sum(dim=-1, keepdim=True) + 1e-8)
        
        # Aggregate values using influence
        output = torch.matmul(influence, V)  # [B, H, N, D]
        output = output.transpose(1, 2).reshape(B, N, C)  # [B, N, C]
        output = self.out_proj(output)
        output = self.dropout(output)
        
        if return_components:
            return {
                'influence_matrix': influence_matrix,
                'output_field': output,
                'head_attention': influence,
                'sparsity_gate': sparsity_weight,
                'directional_bias': directional_bias,
            }
        return influence_matrix, output


class CausalPropagation(nn.Module):
    """Causal propagation operator T for field evolution.
    
    Implements the propagation mechanism from Section 3.3:
    
        F^{t+1}(p_j) = Σ_i G(p_i → p_j) · F^t(p_i)
    
    In matrix form:
        F^{t+1} = G^T · F^t
    
    Multi-step propagation enables:
    - Long-range causal dependency modeling
    - Dynamic evolution of causal effects
    - Counterfactual trajectory simulation
    """
    
    def __init__(
        self,
        feature_dim: int,
        num_steps: int = 3,
        num_heads: int = 8,
        dropout: float = 0.1,
        influence_top_k: Optional[int] = None,
    ):
        super().__init__()
        self.num_steps = num_steps
        self.influence_fn = DirectionalInfluenceFunction(
            feature_dim,
            num_heads=num_heads,
            dropout=dropout,
            top_k=influence_top_k,
        )
        
        # Learnable mixing coefficient alpha in Eq. 6.
        self.mix_logit = nn.Parameter(torch.zeros(1))
        
        # Temporal dynamics (if enabled)
        self.temporal_gate = nn.GRUCell(feature_dim, feature_dim)
    
    def forward(self, 
                field: torch.Tensor, 
                num_steps: Optional[int] = None,
                return_trajectory: bool = False,
                return_details: bool = False) -> Union[torch.Tensor, List[torch.Tensor], Dict[str, Union[torch.Tensor, List[torch.Tensor]]]]:
        """
        Args:
            field: [B, N, C] or [B, H, W, C] initial field state
            num_steps: Number of propagation steps (default: self.num_steps)
            return_trajectory: If True, return all intermediate states
            return_details: If True, return field, trajectory, and per-step influence matrices
            
        Returns:
            final_field: [B, N, C] field after propagation
            or
            trajectory: List[torch.Tensor] all field states if return_trajectory=True
        """
        if field.dim() == 4:
            B, H, W, C = field.shape
            field = field.reshape(B, H * W, C)
            reshape_back = True
            spatial_positions = DirectionalInfluenceFunction._default_positions(
                B, H * W, field.device, field.dtype, H, W
            )
        else:
            B, N, C = field.shape
            reshape_back = False
            spatial_positions = DirectionalInfluenceFunction._default_positions(
                B, N, field.device, field.dtype
            )
        
        steps = num_steps if num_steps is not None else self.num_steps
        
        trajectory = [field]
        influence_matrices = []
        refinement_trajectory = []
        current_field = field
        alpha = torch.sigmoid(self.mix_logit)
        
        for step in range(steps):
            # Compute influence and propagate
            influence_matrix, influence_field = self.influence_fn(current_field, spatial_positions=spatial_positions)
            influence_matrices.append(influence_matrix)
            refinement_trajectory.append(influence_field)
            
            # Propagate: F^{t+1} = G^T · F^t (with residual)
            propagated = torch.bmm(influence_matrix.transpose(1, 2), current_field)
            
            current_field = alpha * propagated + (1 - alpha) * influence_field
            
            trajectory.append(current_field)
        
        def maybe_reshape(x: torch.Tensor) -> torch.Tensor:
            return x.reshape(B, H, W, C) if reshape_back else x

        final_field = maybe_reshape(current_field)

        if return_details:
            return {
                'field': final_field,
                'trajectory': [maybe_reshape(state) for state in trajectory],
                'influence_matrices': influence_matrices,
                'refinement_trajectory': [maybe_reshape(state) for state in refinement_trajectory],
                'mixing_alpha': alpha,
            }

        if return_trajectory:
            return [maybe_reshape(state) for state in trajectory]
        
        return final_field
    
    def propagate_with_intervention(self,
                                    field: torch.Tensor,
                                    intervention_mask: torch.Tensor,
                                    intervention_values: torch.Tensor,
                                    start_step: int = 0,
                                    num_steps: int = 5) -> torch.Tensor:
        """Propagate field with intervention at specific timestep.
        
        Implements counterfactual rollout from Section 3.3:
            F^t → F^t_do → F^{t+k}_do = T^k(F^t_do)
        
        Args:
            field: [B, N, C] initial field state
            intervention_mask: [B, N] binary mask indicating intervention positions
            intervention_values: [B, N, C] values to set at intervention positions
            start_step: Step at which to apply intervention
            num_steps: Total propagation steps after intervention
            
        Returns:
            counterfactual_field: [B, N, C] field after counterfactual propagation
        """
        if field.dim() == 4:
            B, H, W, C = field.shape
            field = field.reshape(B, H * W, C)
            intervention_mask = intervention_mask.reshape(B, H * W)
            intervention_values = intervention_values.reshape(B, H * W, C)
            reshape_back = True
        else:
            reshape_back = False
        
        current_field = field
        
        # Propagate to intervention step
        if start_step > 0:
            current_field = self.forward(current_field, num_steps=start_step)
        
        # Apply intervention: do(F(p) = v)
        mask_expanded = intervention_mask.unsqueeze(-1).float()  # [B, N, 1]
        current_field = mask_expanded * intervention_values + (1 - mask_expanded) * current_field
        
        # Continue propagation
        counterfactual_field = self.forward(current_field, num_steps=num_steps)
        
        if reshape_back:
            counterfactual_field = counterfactual_field.reshape(B, H, W, C)
        
        return counterfactual_field


class InterventionModule(nn.Module):
    """Field-level intervention operations.
    
    Implements the do-operator on causal field from Section 3.2:
    
        F'(p, t) = do(F(p, t))
    
    Supports multiple intervention types:
    - Object removal: Set field values to background/null state
    - Attribute modification: Transform field values
    - Local state change: Modify specific spatial regions
    
    The continuous nature enables fine-grained interventions at arbitrary
    spatial resolutions, unlike discrete variable interventions.
    """
    
    def __init__(self, feature_dim: int, intervention_radius: float = 2.0):
        super().__init__()
        self.feature_dim = feature_dim
        self.intervention_radius = intervention_radius
        
        # Learnable null/background state
        self.register_parameter('null_state', nn.Parameter(torch.randn(feature_dim) * 0.02))
        
        # Attribute transformation network
        self.attribute_transform = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 2),
            nn.LayerNorm(feature_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim * 2, feature_dim)
        )
    
    def create_spatial_mask(self, 
                           center: torch.Tensor, 
                           field_shape: Tuple[int, ...],
                           radius: Optional[float] = None) -> torch.Tensor:
        """Create circular/spherical mask for local intervention.
        
        Args:
            center: [B, 2] center positions (normalized to [0, 1])
            field_shape: (H, W) or (N,) spatial dimensions
            radius: Intervention radius (default: self.intervention_radius)
            
        Returns:
            mask: [B, H, W] or [B, N] binary mask
        """
        if len(field_shape) == 2:
            H, W = field_shape
            N = H * W
            device = center.device
            
            # Create coordinate grid
            y_grid = torch.linspace(0, 1, H, device=device)
            x_grid = torch.linspace(0, 1, W, device=device)
            yy, xx = torch.meshgrid(y_grid, x_grid, indexing='ij')
            coords = torch.stack([xx, yy], dim=-1).reshape(1, N, 2)  # [1, N, 2]
        else:
            N = field_shape[0]
            # Assume 1D position encoding
            coords = torch.linspace(0, 1, N, device=center.device).reshape(1, N, 1)
            center = center[:, :1]  # Use first dim
        
        B = center.shape[0]
        center_expanded = center.unsqueeze(1)  # [B, 1, 2]
        
        # Compute distances
        dists = torch.norm(coords - center_expanded, dim=-1)  # [B, N]
        
        # Create soft mask (Gaussian falloff). Values <= 1 are treated as
        # normalized spatial radii; larger values are interpreted as grid cells.
        r = radius if radius is not None else self.intervention_radius
        sigma = float(r) if float(r) <= 1.0 else float(r) / max(field_shape)
        sigma = max(sigma, 1e-3)
        mask = torch.exp(-dists ** 2 / (2 * sigma ** 2))
        
        if len(field_shape) == 2:
            mask = mask.reshape(B, H, W)
        
        return mask
    
    def remove_object(self, 
                      field: torch.Tensor,
                      position: torch.Tensor,
                      radius: Optional[float] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Remove object by setting field values to null state.
        
        Args:
            field: [B, H, W, C] or [B, N, C] field
            position: [B, 2] center position of object to remove
            radius: Intervention radius
            
        Returns:
            intervened_field: Field after removal
            mask: Intervention mask for tracking
        """
        original_shape = field.shape
        if field.dim() == 4:
            B, H, W, C = field.shape
            field = field.reshape(B, H * W, C)
            N = H * W
            reshape_back = True
        else:
            B, N, C = field.shape
            H = W = int(math.sqrt(N))
            reshape_back = False
        
        # Create mask
        mask = self.create_spatial_mask(position, (H, W), radius)  # [B, H, W]
        mask_flat = mask.reshape(B, N, 1)  # [B, N, 1]
        
        # Intervene: F'(p) = mask * null_state + (1 - mask) * F(p)
        null_expanded = self.null_state.view(1, 1, C).expand(B, N, -1)
        intervened_field = mask_flat * null_expanded + (1 - mask_flat) * field
        
        if reshape_back:
            intervened_field = intervened_field.reshape(original_shape)
        
        return intervened_field, mask
    
    def modify_attribute(self,
                        field: torch.Tensor,
                        position: torch.Tensor,
                        attribute_direction: torch.Tensor,
                        radius: Optional[float] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Modify object attributes by transforming field values.
        
        Args:
            field: [B, H, W, C] or [B, N, C] field
            position: [B, 2] center position
            attribute_direction: [B, C] direction of attribute change in feature space
            radius: Intervention radius
            
        Returns:
            intervened_field: Field after attribute modification
            mask: Intervention mask
        """
        original_shape = field.shape
        if field.dim() == 4:
            B, H, W, C = field.shape
            N = H * W
            field = field.reshape(B, H * W, C)
            reshape_back = True
        else:
            B, N, C = field.shape
            H = W = int(math.sqrt(N))
            reshape_back = False
        
        mask = self.create_spatial_mask(position, (H, W), radius)
        mask_flat = mask.reshape(B, N, 1)
        
        # Transform field values
        transformed = self.attribute_transform(field)  # [B, N, C]
        
        # Apply directional shift
        direction_expanded = attribute_direction.unsqueeze(1)  # [B, 1, C]
        modified = transformed + 0.1 * direction_expanded * field
        
        # Blend based on mask
        intervened_field = mask_flat * modified + (1 - mask_flat) * field
        
        if reshape_back:
            intervened_field = intervened_field.reshape(original_shape)
        
        return intervened_field, mask


class MultimodalCausalField(nn.Module):
    """Complete MetaCausalField module integrating all components.
    
    This is the main interface that combines:
    1. Visual feature extraction (from pretrained encoder)
    2. Continuous causal field construction
    3. Directional influence modeling
    4. Causal propagation
    5. Language-conditioned readout
    
    Replaces the three-stage MLLM-CD pipeline with a unified end-to-end system.
    """
    
    def __init__(self,
                 config: CausalFieldConfig,
                 visual_encoder: Optional[nn.Module] = None,
                 vocab_size: Optional[int] = None,
                 num_factors: int = 0,
                 enable_language: bool = True):
        super().__init__()
        self.config = config
        self.vocab_size = int(vocab_size or config.vocab_size)
        self.num_factors = int(num_factors)
        self.enable_language = enable_language
        
        # Visual encoder (e.g., ViT backbone)
        self.visual_encoder = visual_encoder
        
        # Field construction
        self.interpolation = GaussianInterpolation(sigma=config.gaussian_sigma)
        
        # Feature projection to causal field space
        self.field_projection = nn.Sequential(
            nn.Linear(config.feature_dim, config.feature_dim),
            nn.LayerNorm(config.feature_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.feature_dim, config.feature_dim)
        )
        
        # Causal propagation
        self.propagation = CausalPropagation(
            config.feature_dim,
            config.num_propagation_steps,
            num_heads=config.num_heads,
            dropout=config.dropout,
            influence_top_k=config.influence_top_k,
        )
        
        # Intervention module
        self.intervention = InterventionModule(config.feature_dim, config.intervention_radius)
        
        # Language fusion (cross-attention)
        self.language_fusion = nn.MultiheadAttention(
            embed_dim=config.feature_dim,
            num_heads=config.num_heads,
            dropout=config.dropout,
            batch_first=True
        )

        if enable_language:
            self.text_encoder = TextEncoder(
                vocab_size=self.vocab_size,
                feature_dim=config.feature_dim,
                max_length=config.max_text_length,
                dropout=config.dropout,
            )
            self.text_decoder = TextCausalDecoder(
                vocab_size=self.vocab_size,
                feature_dim=config.feature_dim,
                max_length=config.max_text_length,
                dropout=config.dropout,
            )
        else:
            self.text_encoder = None
            self.text_decoder = None
        
        # Output readout
        self.readout = nn.Sequential(
            nn.Linear(config.feature_dim, config.feature_dim),
            nn.LayerNorm(config.feature_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.feature_dim, config.feature_dim)
        )

        self.score_head = nn.Linear(config.feature_dim, 1)
        self.factor_head = nn.Linear(config.feature_dim, self.num_factors * config.num_factor_classes) \
            if self.num_factors > 0 else None
        self.factor_localizer = nn.Linear(config.feature_dim, self.num_factors) if self.num_factors > 0 else None

    def factor_attention_from_field(self, field: torch.Tensor) -> Optional[torch.Tensor]:
        """Return factor-to-position attention [B, F, N] for factor-level graphs."""
        if self.factor_localizer is None:
            return None
        if field.dim() == 4:
            B, H, W, C = field.shape
            field_flat = field.reshape(B, H * W, C)
        else:
            field_flat = field
        logits = self.factor_localizer(field_flat)  # [B, N, F]
        return torch.softmax(logits.transpose(1, 2), dim=-1)

    @staticmethod
    def factor_influence_from_attention(
        influence_matrix: torch.Tensor,
        factor_attention: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        """Aggregate patch-level influence G into factor-level A^T G A."""
        if factor_attention is None:
            return None
        factor_matrix = torch.bmm(
            torch.bmm(factor_attention, influence_matrix),
            factor_attention.transpose(1, 2),
        )
        eye = torch.eye(factor_matrix.shape[-1], device=factor_matrix.device, dtype=torch.bool).unsqueeze(0)
        factor_matrix = factor_matrix.masked_fill(eye, 0.0)
        return factor_matrix

    def infer_intervention_position(
        self,
        field: torch.Tensor,
        target_factor_idx: Optional[int] = None,
    ) -> torch.Tensor:
        """Infer a normalized intervention center from factor attention or field saliency."""
        B, H, W, C = field.shape
        device = field.device
        y_grid = torch.linspace(0, 1, H, device=device, dtype=field.dtype)
        x_grid = torch.linspace(0, 1, W, device=device, dtype=field.dtype)
        yy, xx = torch.meshgrid(y_grid, x_grid, indexing='ij')
        coords = torch.stack([xx, yy], dim=-1).reshape(1, H * W, 2)

        factor_attention = self.factor_attention_from_field(field)
        if factor_attention is not None and target_factor_idx is not None:
            idx = int(target_factor_idx)
            if 0 <= idx < factor_attention.shape[1]:
                weights = factor_attention[:, idx]
                return torch.bmm(weights.unsqueeze(1), coords.expand(B, -1, -1)).squeeze(1)

        saliency = field.abs().mean(dim=-1).reshape(B, H * W)
        weights = torch.softmax(saliency, dim=-1)
        return torch.bmm(weights.unsqueeze(1), coords.expand(B, -1, -1)).squeeze(1)

    def encode_visual_input(self, visual_input: torch.Tensor) -> torch.Tensor:
        """Accept either patch features or raw image/video tensors."""
        if visual_input.dim() in {4, 5} and self.visual_encoder is not None:
            return self.visual_encoder(visual_input)
        return visual_input
    
    def build_causal_field(self, 
                          visual_features: torch.Tensor,
                          patch_positions: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Build initial causal field from visual features.
        
        Args:
            visual_features: [B, N, C] patch features from visual encoder
            patch_positions: [B, N, 2] patch positions (optional)
            
        Returns:
            field: [B, H, W, C] causal field
        """
        visual_features = self.encode_visual_input(visual_features)
        B, N, C = visual_features.shape
        source_h = max(1, int(math.sqrt(N)))
        source_w = int(math.ceil(N / source_h))
        
        if patch_positions is None:
            # Generate default grid positions
            device = visual_features.device
            y_grid = torch.linspace(0, 1, source_h, device=device)
            x_grid = torch.linspace(0, 1, source_w, device=device)
            yy, xx = torch.meshgrid(y_grid, x_grid, indexing='ij')
            patch_positions = torch.stack([xx, yy], dim=-1).reshape(1, source_h * source_w, 2)[:, :N]
            patch_positions = patch_positions.expand(B, -1, -1)
        
        # Project features to field space
        projected = self.field_projection(visual_features)
        
        # Build continuous field via interpolation
        H, W = self.config.spatial_size
        device = visual_features.device
        y_query = torch.linspace(0, 1, H, device=device)
        x_query = torch.linspace(0, 1, W, device=device)
        yy, xx = torch.meshgrid(y_query, x_query, indexing='ij')
        query_positions = torch.stack([xx, yy], dim=-1).reshape(1, H * W, 2).expand(B, -1, -1)
        field = self.interpolation(projected, patch_positions, query_positions=query_positions)
        
        return field
    
    def forward(self, 
                visual_features: torch.Tensor,
                language_tokens: Optional[torch.Tensor] = None,
                input_ids: Optional[torch.Tensor] = None,
                decoder_input_ids: Optional[torch.Tensor] = None,
                patch_positions: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Forward pass through MetaCausalField.
        
        Args:
            visual_features: [B, N, C] visual patch features
            language_tokens: [B, L, C] language features (optional)
            patch_positions: [B, N, 2] spatial positions
            
        Returns:
            outputs: Dict containing:
                - 'field': Final causal field [B, H, W, C]
                - 'influence_matrix': Learned causal structure [B, N, N]
                - 'readout': Output representation [B, C]
        """
        visual_features = self.encode_visual_input(visual_features)

        # Build initial field
        field = self.build_causal_field(visual_features, patch_positions)
        B, H, W, C = field.shape
        N = H * W
        
        # Propagate causal effects
        propagation_details = self.propagation(field, return_details=True)
        field_propagated = propagation_details['field']
        influence_matrices = propagation_details['influence_matrices']
        if influence_matrices:
            influence_matrix = influence_matrices[-1]
        else:
            influence_matrix, _ = self.propagation.influence_fn(field_propagated.reshape(B, N, C))
        
        if language_tokens is None and input_ids is not None and self.text_encoder is not None:
            language_tokens = self.text_encoder(input_ids)

        # Fuse with language if provided
        if language_tokens is not None:
            field_flat = field_propagated.reshape(B, N, C)
            key_padding_mask = (
                input_ids.eq(self.config.pad_token_id)
                if input_ids is not None and input_ids.shape[1] == language_tokens.shape[1]
                else None
            )
            # Cross-attention: field attends to language
            fused, _ = self.language_fusion(
                query=field_flat,
                key=language_tokens,
                value=language_tokens,
                key_padding_mask=key_padding_mask,
            )
            field_fused = fused.reshape(B, H, W, C)
        else:
            field_fused = field_propagated
        
        # Global readout
        field_pooled = field_fused.mean(dim=[1, 2])  # [B, C]
        output = self.readout(field_pooled)

        memory = field_fused.reshape(B, N, C)
        score_pred = self.score_head(output).squeeze(-1)
        factor_logits = None
        if self.factor_head is not None:
            factor_logits = self.factor_head(output).reshape(
                B, self.num_factors, self.config.num_factor_classes
            )
        factor_attention = self.factor_attention_from_field(field_fused)
        factor_influence_matrix = self.factor_influence_from_attention(influence_matrix, factor_attention)

        lm_logits = None
        if decoder_input_ids is not None and self.text_decoder is not None:
            lm_logits = self.text_decoder(decoder_input_ids, memory)
        
        outputs = {
            'field': field_fused,
            'influence_matrix': influence_matrix,
            'influence_matrices': influence_matrices,
            'readout': output,
            'initial_field': field,
            'field_trajectory': propagation_details['trajectory'],
            'mixing_alpha': propagation_details['mixing_alpha'],
            'score_pred': score_pred,
        }
        if factor_logits is not None:
            outputs['factor_logits'] = factor_logits
        if factor_attention is not None:
            outputs['factor_spatial_attention'] = factor_attention
            outputs['factor_influence_matrix'] = factor_influence_matrix
        if lm_logits is not None:
            outputs['lm_logits'] = lm_logits
        return outputs
    
    def counterfactual_forward(self,
                               visual_features: torch.Tensor,
                               intervention_type: str,
                               intervention_params: Dict,
                               language_tokens: Optional[torch.Tensor] = None,
                               input_ids: Optional[torch.Tensor] = None,
                               decoder_input_ids: Optional[torch.Tensor] = None,
                               num_rollout_steps: int = 5) -> Dict[str, torch.Tensor]:
        """Generate counterfactual predictions via field intervention.
        
        Args:
            visual_features: [B, N, C] visual features
            intervention_type: Type of intervention ('remove', 'modify', 'custom')
            intervention_params: Parameters for intervention
            language_tokens: Optional language conditioning
            num_rollout_steps: Number of propagation steps after intervention
            
        Returns:
            outputs: Dict with factual and counterfactual outputs
        """
        visual_features = self.encode_visual_input(visual_features)

        # Build initial field
        field = self.build_causal_field(visual_features)
        B, H, W, C = field.shape
        
        # Factual propagation
        factual_details = self.propagation(field, return_details=True)
        field_factual = factual_details['field']
        
        # Apply intervention
        if intervention_type == 'remove':
            if 'position' not in intervention_params or intervention_params['position'] is None:
                intervention_params = dict(intervention_params)
                intervention_params['position'] = self.infer_intervention_position(
                    field,
                    intervention_params.get('target_factor_idx'),
                )
            field_intervened, mask = self.intervention.remove_object(
                field, 
                intervention_params['position'],
                intervention_params.get('radius')
            )
        elif intervention_type == 'modify':
            if 'position' not in intervention_params or intervention_params['position'] is None:
                intervention_params = dict(intervention_params)
                intervention_params['position'] = self.infer_intervention_position(
                    field,
                    intervention_params.get('target_factor_idx'),
                )
            field_intervened, mask = self.intervention.modify_attribute(
                field,
                intervention_params['position'],
                intervention_params['direction'],
                intervention_params.get('radius')
            )
        else:
            value = intervention_params['value'].to(device=field.device, dtype=field.dtype)
            mask = intervention_params.get('mask')
            if mask is None:
                mask = torch.ones(B, H, W, device=field.device, dtype=field.dtype)
            else:
                mask = mask.to(device=field.device, dtype=field.dtype)
                if mask.dim() == 2:
                    mask = mask.reshape(B, H, W)

            if value.dim() == 2:
                value = value.view(B, 1, 1, C).expand(B, H, W, C)
            elif value.dim() == 3:
                value = value.reshape(B, H, W, C)

            mask_flat = mask.reshape(B, H * W, 1)
            field_intervened = (
                mask_flat * value.reshape(B, H * W, C)
                + (1 - mask_flat) * field.reshape(B, H * W, C)
            ).reshape(B, H, W, C)
        
        # Counterfactual rollout
        counterfactual_details = self.propagation(
            field_intervened,
            num_steps=num_rollout_steps,
            return_details=True,
        )
        field_counterfactual = counterfactual_details['field']
        raw_field_factual = field_factual
        raw_field_counterfactual = field_counterfactual

        if language_tokens is None and input_ids is not None and self.text_encoder is not None:
            language_tokens = self.text_encoder(input_ids)

        if language_tokens is not None:
            key_padding_mask = (
                input_ids.eq(self.config.pad_token_id)
                if input_ids is not None and input_ids.shape[1] == language_tokens.shape[1]
                else None
            )
            field_flat = field_factual.reshape(B, H * W, C)
            factual_fused, _ = self.language_fusion(
                field_flat,
                language_tokens,
                language_tokens,
                key_padding_mask=key_padding_mask,
            )
            field_factual = factual_fused.reshape(B, H, W, C)

            cf_flat = field_counterfactual.reshape(B, H * W, C)
            cf_fused, _ = self.language_fusion(
                cf_flat,
                language_tokens,
                language_tokens,
                key_padding_mask=key_padding_mask,
            )
            field_counterfactual = cf_fused.reshape(B, H, W, C)
        
        # Decode both
        output_factual = self.readout(field_factual.mean(dim=[1, 2]))
        output_counterfactual = self.readout(field_counterfactual.mean(dim=[1, 2]))
        score_factual = self.score_head(output_factual).squeeze(-1)
        score_counterfactual = self.score_head(output_counterfactual).squeeze(-1)

        factor_logits_factual = None
        factor_logits_counterfactual = None
        if self.factor_head is not None:
            factor_logits_factual = self.factor_head(output_factual).reshape(
                B, self.num_factors, self.config.num_factor_classes
            )
            factor_logits_counterfactual = self.factor_head(output_counterfactual).reshape(
                B, self.num_factors, self.config.num_factor_classes
            )
        factual_factor_attention = self.factor_attention_from_field(field_factual)
        counterfactual_factor_attention = self.factor_attention_from_field(field_counterfactual)
        factual_factor_influence = None
        counterfactual_factor_influence = None
        if factual_factor_attention is not None:
            factual_influence = factual_details['influence_matrices'][-1] if factual_details['influence_matrices'] else None
            counterfactual_influence = (
                counterfactual_details['influence_matrices'][-1]
                if counterfactual_details['influence_matrices']
                else None
            )
            if factual_influence is not None:
                factual_factor_influence = self.factor_influence_from_attention(
                    factual_influence,
                    factual_factor_attention,
                )
            if counterfactual_influence is not None:
                counterfactual_factor_influence = self.factor_influence_from_attention(
                    counterfactual_influence,
                    counterfactual_factor_attention,
                )

        lm_logits_counterfactual = None
        if decoder_input_ids is not None and self.text_decoder is not None:
            lm_logits_counterfactual = self.text_decoder(
                decoder_input_ids,
                field_counterfactual.reshape(B, H * W, C),
            )
        
        delta_trajectory = [
            cf_state - factual_state
            for factual_state, cf_state in zip(
                factual_details['trajectory'],
                counterfactual_details['trajectory'],
            )
        ]

        outputs = {
            'field_factual': field_factual,
            'field_counterfactual': field_counterfactual,
            'field_factual_unfused': raw_field_factual,
            'field_counterfactual_unfused': raw_field_counterfactual,
            'factual_trajectory': factual_details['trajectory'],
            'counterfactual_trajectory': counterfactual_details['trajectory'],
            'cf_pred_trajectory': counterfactual_details['trajectory'][1:],
            'delta_trajectory': delta_trajectory,
            'factual_influence_matrices': factual_details['influence_matrices'],
            'counterfactual_influence_matrices': counterfactual_details['influence_matrices'],
            'output_factual': output_factual,
            'output_counterfactual': output_counterfactual,
            'score_factual': score_factual,
            'score_counterfactual': score_counterfactual,
            'counterfactual_effect': raw_field_counterfactual - raw_field_factual,
            'intervention_mask': mask,
            'mixing_alpha': counterfactual_details['mixing_alpha'],
        }
        if factor_logits_factual is not None:
            outputs['factor_logits_factual'] = factor_logits_factual
            outputs['factor_logits_counterfactual'] = factor_logits_counterfactual
        if factual_factor_attention is not None:
            outputs['factor_spatial_attention_factual'] = factual_factor_attention
            outputs['factor_spatial_attention_counterfactual'] = counterfactual_factor_attention
            outputs['factor_influence_matrix_factual'] = factual_factor_influence
            outputs['factor_influence_matrix_counterfactual'] = counterfactual_factor_influence
        if lm_logits_counterfactual is not None:
            outputs['lm_logits_counterfactual'] = lm_logits_counterfactual
        return outputs


# ==============================================================================
# Loss Functions for MetaCausalField Training
# ==============================================================================

class MetaCausalLoss(nn.Module):
    """Multi-objective loss for MetaCausalField training (Section 3.4).
    
    Combines:
    1. Language modeling loss (basic capability)
    2. Contrastive loss for factor discovery (implicit factor identification)
    3. Causal consistency loss (invariance principle)
    4. Counterfactual evolution loss (dynamic mechanism learning)
    5. Structure regularization (sparsity & smoothness)
    
    Total loss: L = L_lm + λ_1 * L_contrast + λ_2 * L_inv + λ_3 * L_cf + λ_4 * L_reg
    """
    
    def __init__(self, config: CausalFieldConfig):
        super().__init__()
        self.config = config
        
        # Learnable temperature for contrastive loss
        self.temperature = nn.Parameter(torch.tensor(0.07))
        
        # Loss weights
        self.lambda_contrast = config.lambda_sparsity
        self.lambda_consistency = config.lambda_consistency
        self.lambda_counterfactual = config.lambda_counterfactual
        self.lambda_reg = config.lambda_sparsity
    
    def contrastive_loss(self, 
                         field_a: torch.Tensor, 
                         field_b: torch.Tensor) -> torch.Tensor:
        """Contrastive loss for implicit factor discovery.
        
        Replaces explicit factor discovery in MLLM-CD by identifying
        regions with significant field differences:
        
        L_contrast = Σ_p |F_a(p) - F_b(p)|
        
        High-difference regions ≈ causal factors
        """
        if field_a.dim() == 4:
            # [B, H, W, C] -> [B, N, C]
            B, H, W, C = field_a.shape
            field_a = field_a.reshape(B, H * W, C)
            field_b = field_b.reshape(B, H * W, C)
        
        # Compute spatial difference
        diff = torch.norm(field_a - field_b, p=2, dim=-1)  # [B, N]
        
        # Contrastive loss: maximize difference for contrastive pairs
        loss = -torch.mean(diff)
        
        return loss
    
    def causal_consistency_loss(self, 
                                 field1: torch.Tensor, 
                                 field2: torch.Tensor) -> torch.Tensor:
        """Invariance-based consistency loss.
        
        Based on the principle that causal mechanisms should be invariant
        across different environments/observations.
        
        L_inv = ||F_1 - F_2||
        
        Encourages the model to learn stable causal representations
        that are robust to spurious correlations.
        """
        if field1.dim() == 4:
            field1 = field1.flatten(1)
            field2 = field2.flatten(1)
        
        return F.mse_loss(field1, field2)
    
    def counterfactual_evolution_loss(self,
                                       predicted_trajectory: List[torch.Tensor],
                                       target_trajectory: List[torch.Tensor]) -> torch.Tensor:
        """Dynamic counterfactual prediction loss (core innovation).
        
        Unlike MLLM-CD which uses counterfactuals only for structure refinement,
        we directly learn dynamic causal mechanisms:
        
        L_cf = Σ_k ||F̂^{t+k} - F^{t+k}||
        """
        if isinstance(predicted_trajectory, torch.Tensor):
            predicted_trajectory = list(predicted_trajectory.unbind(0))
        if isinstance(target_trajectory, torch.Tensor):
            target_trajectory = list(target_trajectory.unbind(0))
        if not predicted_trajectory or not target_trajectory:
            device = self.temperature.device
            return torch.zeros((), device=device)
        loss = 0.0
        for pred, target in zip(predicted_trajectory, target_trajectory):
            loss = loss + F.mse_loss(pred, target)
        return loss / min(len(predicted_trajectory), len(target_trajectory))

    def _field_smoothness(self, field: torch.Tensor) -> torch.Tensor:
        if field.dim() == 4:
            grad_h = field[:, 1:, :, :] - field[:, :-1, :, :]
            grad_w = field[:, :, 1:, :] - field[:, :, :-1, :]
            return (grad_h ** 2).mean() + (grad_w ** 2).mean()
        if field.dim() == 3:
            B, N, C = field.shape
            H = int(math.sqrt(N))
            if H * H == N:
                return self._field_smoothness(field.reshape(B, H, H, C))
        return torch.zeros((), device=field.device, dtype=field.dtype)
    
    def structure_regularization(self,
                                  influence_matrix: torch.Tensor,
                                  field: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Structure regularization losses.
        
        Sparsity: L1 penalty on influence matrix (encourages sparse causal structure)
        Smoothness: Gradient penalty on field (encourages spatial continuity)
        
        L_reg = λ_1 |G|_1 + λ_2 |∇F|²
        """
        if isinstance(influence_matrix, (list, tuple)):
            matrices = [m for m in influence_matrix if m is not None]
            if matrices:
                sparsity_loss = torch.stack([m.abs().mean() for m in matrices]).mean()
            else:
                device = field[0].device if isinstance(field, (list, tuple)) else field.device
                sparsity_loss = torch.zeros((), device=device)
        else:
            sparsity_loss = influence_matrix.abs().mean()

        if isinstance(field, (list, tuple)):
            states = [state for state in field if state is not None]
            if states:
                smoothness_loss = torch.stack([self._field_smoothness(state) for state in states]).mean()
            else:
                smoothness_loss = torch.zeros((), device=sparsity_loss.device, dtype=sparsity_loss.dtype)
        else:
            smoothness_loss = self._field_smoothness(field)
        
        return sparsity_loss, smoothness_loss
    
    def forward(self,
                outputs: Dict[str, torch.Tensor],
                targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute combined loss.
        
        L = L_lm + λ_1 L_inv + λ_2 L_cf + λ_3 L_reg
        """
        losses = {}
        
        # Language modeling loss (if provided)
        if 'lm_logits' in outputs and 'lm_targets' in targets:
            losses['lm'] = F.cross_entropy(
                outputs['lm_logits'].reshape(-1, outputs['lm_logits'].size(-1)),
                targets['lm_targets'].reshape(-1),
                ignore_index=self.config.pad_token_id,
            )

        if 'score_pred' in outputs and 'score_targets' in targets:
            losses['score'] = F.mse_loss(outputs['score_pred'], targets['score_targets'].float())

        if 'factor_logits' in outputs and 'factor_targets' in targets:
            logits = outputs['factor_logits'].reshape(-1, self.config.num_factor_classes)
            factor_targets = targets['factor_targets'].reshape(-1).long()
            losses['factor'] = F.cross_entropy(logits, factor_targets, ignore_index=-100)
        
        # Causal consistency
        if 'field_env1' in outputs and 'field_env2' in outputs:
            losses['consistency'] = self.causal_consistency_loss(
                outputs['field_env1'], outputs['field_env2']
            )
        
        # Counterfactual evolution
        if 'cf_pred_trajectory' in outputs and 'cf_target_trajectory' in targets:
            losses['counterfactual'] = self.counterfactual_evolution_loss(
                outputs['cf_pred_trajectory'], targets['cf_target_trajectory']
            )
        elif 'field_counterfactual_unfused' in outputs and 'cf_field_targets' in targets:
            losses['counterfactual'] = F.mse_loss(
                outputs['field_counterfactual_unfused'],
                targets['cf_field_targets'],
            )

        if 'score_counterfactual' in outputs and 'cf_score_targets' in targets:
            losses['counterfactual_score'] = F.mse_loss(
                outputs['score_counterfactual'], targets['cf_score_targets'].float()
            )

        if 'lm_logits_counterfactual' in outputs and 'cf_lm_targets' in targets:
            losses['counterfactual_lm'] = F.cross_entropy(
                outputs['lm_logits_counterfactual'].reshape(-1, outputs['lm_logits_counterfactual'].size(-1)),
                targets['cf_lm_targets'].reshape(-1),
                ignore_index=self.config.pad_token_id,
            )
        
        # Structure regularization
        influence_for_reg = outputs.get('influence_matrices', outputs.get('influence_matrix'))
        field_for_reg = outputs.get('field_trajectory', outputs.get('field'))
        if influence_for_reg is not None and field_for_reg is not None:
            sparsity, smoothness = self.structure_regularization(
                influence_for_reg, field_for_reg
            )
            losses['sparsity'] = sparsity
            losses['smoothness'] = smoothness
        
        # Total loss
        total = losses.get('lm', 0) + \
                losses.get('score', 0) + \
                losses.get('factor', 0) + \
                self.lambda_consistency * losses.get('consistency', 0) + \
                self.lambda_counterfactual * losses.get('counterfactual', 0) + \
                self.lambda_counterfactual * losses.get('counterfactual_score', 0) + \
                self.lambda_counterfactual * losses.get('counterfactual_lm', 0) + \
                self.config.lambda_sparsity * losses.get('sparsity', 0) + \
                self.config.lambda_smoothness * losses.get('smoothness', 0)
        
        losses['total'] = total
        
        return losses


# ==============================================================================
# Utility Functions for Analysis and Visualization
# ==============================================================================

def extract_causal_graph(influence_matrix: torch.Tensor,
                         threshold: float = 0.1) -> np.ndarray:
    """Extract discrete causal graph from continuous influence matrix.
    
    Args:
        influence_matrix: [N, N] influence weights
        threshold: Minimum influence to consider as edge
        
    Returns:
        adj_matrix: [N, N] binary adjacency matrix
    """
    if isinstance(influence_matrix, torch.Tensor):
        influence_matrix = influence_matrix.detach().cpu().numpy()
    
    # Threshold to get binary graph
    adj_matrix = (influence_matrix > threshold).astype(np.int32)
    
    # Remove self-loops
    np.fill_diagonal(adj_matrix, 0)
    
    return adj_matrix


def compute_causal_effects(field: torch.Tensor,
                          influence_matrix: torch.Tensor,
                          intervention_position: int) -> torch.Tensor:
    """Compute causal effect propagation from an intervention.
    
    Args:
        field: [N, C] field state
        influence_matrix: [N, N] influence weights
        intervention_position: Index of intervened position
        
    Returns:
        effects: [N] causal effect strength at each position
    """
    # Effect is the accumulated influence from intervention position
    effects = influence_matrix[:, intervention_position]  # [N]
    
    # Normalize
    effects = effects / (effects.sum() + 1e-8)
    
    return effects


def visualize_causal_field(field: torch.Tensor,
                          influence_matrix: Optional[torch.Tensor] = None,
                          save_path: Optional[str] = None):
    """Visualize causal field and influence structure.
    
    Args:
        field: [H, W, C] or [N, C] field values
        influence_matrix: [N, N] optional influence matrix
        save_path: Path to save visualization
    """
    import matplotlib.pyplot as plt
    
    if field.dim() == 3:
        H, W, C = field.shape
        field_2d = field.mean(dim=-1).cpu().numpy()  # Average over channels
    else:
        N, C = field.shape
        H = W = int(math.sqrt(N))
        field_2d = field.mean(dim=-1).cpu().numpy().reshape(H, W)
    
    fig, axes = plt.subplots(1, 2 if influence_matrix is None else 3, 
                            figsize=(12 if influence_matrix is None else 18, 5))
    
    # Field intensity
    im1 = axes[0].imshow(field_2d, cmap='viridis')
    axes[0].set_title('Causal Field Intensity')
    axes[0].axis('off')
    plt.colorbar(im1, ax=axes[0])
    
    # Field variance (across channels)
    if field.dim() == 3:
        field_var = field.var(dim=-1).cpu().numpy()
    else:
        field_var = field.var(dim=-1).cpu().numpy().reshape(H, W)
    
    im2 = axes[1].imshow(field_var, cmap='hot')
    axes[1].set_title('Field Variance (Uncertainty)')
    axes[1].axis('off')
    plt.colorbar(im2, ax=axes[1])
    
    # Influence matrix
    if influence_matrix is not None:
        im3 = axes[2].imshow(influence_matrix.cpu().numpy(), cmap='Blues')
        axes[2].set_title('Causal Influence Matrix')
        axes[2].set_xlabel('From Position')
        axes[2].set_ylabel('To Position')
        plt.colorbar(im3, ax=axes[2])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()
