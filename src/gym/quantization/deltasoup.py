"""
DeltaSoup: Community-Driven Model Improvement Aggregation
Copyright 2025 Zoo Labs Foundation Inc.

Byzantine-robust aggregation of personalized model improvements
for community-driven model evolution.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
from dataclasses import dataclass
from enum import Enum
import hashlib
import json


class AggregationMethod(Enum):
    """Aggregation methods for DeltaSoup"""
    MEAN = "mean"
    MEDIAN = "median"
    TRIMMED_MEAN = "trimmed_mean"
    WEIGHTED_MEAN = "weighted_mean"
    BYZANTINE_ROBUST = "byzantine_robust"
    FEDERATED_AVERAGE = "federated_average"
    KRUM = "krum"
    MULTI_KRUM = "multi_krum"


@dataclass
class DeltaSoupConfig:
    """Configuration for DeltaSoup aggregation"""
    
    # Aggregation settings
    method: AggregationMethod = AggregationMethod.BYZANTINE_ROBUST
    trim_percent: float = 0.2  # For trimmed mean
    byzantine_threshold: float = 0.3  # Max fraction of byzantine clients
    
    # Privacy settings
    differential_privacy: bool = False
    privacy_epsilon: float = 1.0  # Privacy budget
    privacy_delta: float = 1e-5  # Privacy parameter
    noise_scale: float = 0.01  # Noise scale for DP
    
    # Validation settings
    validate_contributions: bool = True
    quality_threshold: float = 0.8  # Min quality score
    diversity_bonus: float = 0.1  # Bonus for diverse contributions
    
    # Community settings
    min_contributors: int = 3  # Minimum contributors for aggregation
    max_contributors: int = 1000  # Maximum contributors
    contribution_window: int = 86400  # Time window in seconds (24h)
    
    # Compression settings
    use_bitdelta: bool = True  # Use BitDelta compression
    use_deltaquant: bool = False  # Use DeltaQuant compression
    compression_threshold: float = 0.9  # Min compression ratio
    
    # Reward settings
    enable_rewards: bool = True  # Enable contributor rewards
    reward_pool: float = 1000.0  # Total reward pool
    quality_weight: float = 0.7  # Weight for quality in rewards
    participation_weight: float = 0.3  # Weight for participation


class ContributorProfile:
    """Profile for a DeltaSoup contributor"""
    
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.contribution_count = 0
        self.quality_scores: List[float] = []
        self.reputation_score = 1.0
        self.total_rewards = 0.0
        self.last_contribution = None
        self.contribution_hashes: List[str] = []
    
    def update_reputation(self, quality_score: float):
        """Update reputation based on contribution quality"""
        self.quality_scores.append(quality_score)
        # Exponential moving average
        alpha = 0.1
        self.reputation_score = (1 - alpha) * self.reputation_score + alpha * quality_score
        self.contribution_count += 1
    
    def get_weight(self) -> float:
        """Get aggregation weight based on reputation"""
        return self.reputation_score
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert profile to dictionary"""
        return {
            'user_id': self.user_id,
            'contribution_count': self.contribution_count,
            'reputation_score': self.reputation_score,
            'total_rewards': self.total_rewards,
            'average_quality': np.mean(self.quality_scores) if self.quality_scores else 0
        }


class DeltaSoup:
    """
    DeltaSoup: Aggregate community model improvements with Byzantine robustness.
    Supports privacy-preserving aggregation and contributor rewards.
    """
    
    def __init__(self, config: DeltaSoupConfig):
        self.config = config
        self.contributors: Dict[str, ContributorProfile] = {}
        self.aggregated_deltas: Dict[str, torch.Tensor] = {}
        self.contribution_buffer: List[Dict[str, Any]] = []
        
        # Initialize quantizers if enabled
        if config.use_bitdelta:
            from .bitdelta import BitDeltaQuantizer, BitDeltaConfig
            self.bitdelta_quantizer = BitDeltaQuantizer(BitDeltaConfig())
        
        if config.use_deltaquant:
            from .deltaquant import DeltaQuantizer, DeltaQuantConfig
            self.deltaquant_quantizer = DeltaQuantizer(DeltaQuantConfig())
    
    def contribute(
        self,
        user_id: str,
        model: nn.Module,
        base_model: nn.Module,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Contribute a fine-tuned model to DeltaSoup.
        
        Args:
            user_id: Unique contributor identifier
            model: Fine-tuned model
            base_model: Base model
            metadata: Optional metadata about the contribution
            
        Returns:
            Contribution hash for tracking
        """
        # Create or get contributor profile
        if user_id not in self.contributors:
            self.contributors[user_id] = ContributorProfile(user_id)
        
        profile = self.contributors[user_id]
        
        # Extract and compress deltas
        deltas = self._extract_deltas(model, base_model)
        
        # Validate contribution if enabled
        if self.config.validate_contributions:
            quality_score = self._validate_contribution(deltas, metadata)
            if quality_score < self.config.quality_threshold:
                return None
            profile.update_reputation(quality_score)
        
        # Create contribution hash
        contribution_hash = self._hash_contribution(user_id, deltas)
        profile.contribution_hashes.append(contribution_hash)
        
        # Add to buffer
        self.contribution_buffer.append({
            'user_id': user_id,
            'deltas': deltas,
            'metadata': metadata,
            'timestamp': torch.tensor(np.datetime64('now').astype(float)),
            'quality_score': quality_score if self.config.validate_contributions else 1.0,
            'hash': contribution_hash
        })
        
        print(f"Contribution {contribution_hash[:8]} accepted from {user_id}")
        return contribution_hash
    
    def aggregate(
        self,
        min_contributors: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Aggregate all contributions in the buffer.
        
        Args:
            min_contributors: Minimum contributors required
            
        Returns:
            Aggregated model deltas
        """
        min_contributors = min_contributors or self.config.min_contributors
        
        if len(self.contribution_buffer) < min_contributors:
            print(f"Not enough contributors ({len(self.contribution_buffer)}/{min_contributors})")
            return None
        
        # Group contributions by layer
        layer_contributions = self._group_by_layer()
        
        # Aggregate each layer
        aggregated = {}
        for layer_name, contributions in layer_contributions.items():
            if self.config.method == AggregationMethod.BYZANTINE_ROBUST:
                aggregated[layer_name] = self._byzantine_robust_aggregate(contributions)
            elif self.config.method == AggregationMethod.KRUM:
                aggregated[layer_name] = self._krum_aggregate(contributions)
            elif self.config.method == AggregationMethod.TRIMMED_MEAN:
                aggregated[layer_name] = self._trimmed_mean_aggregate(contributions)
            elif self.config.method == AggregationMethod.MEDIAN:
                aggregated[layer_name] = self._median_aggregate(contributions)
            elif self.config.method == AggregationMethod.WEIGHTED_MEAN:
                aggregated[layer_name] = self._weighted_mean_aggregate(contributions)
            else:
                aggregated[layer_name] = self._mean_aggregate(contributions)
        
        # Add differential privacy if enabled
        if self.config.differential_privacy:
            aggregated = self._add_differential_privacy(aggregated)
        
        # Distribute rewards if enabled
        if self.config.enable_rewards:
            self._distribute_rewards()
        
        # Store aggregated deltas
        self.aggregated_deltas = aggregated
        
        # Clear buffer after aggregation
        self.contribution_buffer.clear()
        
        return aggregated
    
    def _extract_deltas(
        self,
        model: nn.Module,
        base_model: nn.Module
    ) -> Dict[str, torch.Tensor]:
        """Extract and optionally compress model deltas."""
        deltas = {}
        
        for name, param in model.named_parameters():
            if name in dict(base_model.named_parameters()):
                base_param = dict(base_model.named_parameters())[name]
                delta = param.data - base_param.data
                
                # Compress if enabled
                if self.config.use_bitdelta:
                    signs, scales = self.bitdelta_quantizer.quantize_delta(
                        param.data, base_param.data, name
                    )
                    delta = self.bitdelta_quantizer.dequantize_delta(
                        signs, scales, param.shape
                    )
                elif self.config.use_deltaquant:
                    quantized, params = self.deltaquant_quantizer.quantize_delta(
                        param.data, base_param.data, name
                    )
                    delta = self.deltaquant_quantizer.dequantize_delta(
                        quantized, params, param.shape
                    )
                
                deltas[name] = delta
        
        return deltas
    
    def _validate_contribution(
        self,
        deltas: Dict[str, torch.Tensor],
        metadata: Optional[Dict[str, Any]] = None
    ) -> float:
        """Validate contribution quality."""
        quality_scores = []
        
        # Check delta magnitudes (not too large/small)
        for name, delta in deltas.items():
            magnitude = delta.abs().mean().item()
            if magnitude < 1e-6:
                quality_scores.append(0.0)  # Too small
            elif magnitude > 10.0:
                quality_scores.append(0.0)  # Too large
            else:
                # Normalize to [0, 1]
                quality_scores.append(1.0 - abs(np.log10(magnitude)) / 3.0)
        
        # Check diversity if metadata provided
        if metadata and 'training_data' in metadata:
            diversity_score = self._calculate_diversity(metadata['training_data'])
            quality_scores.append(diversity_score)
        
        return np.mean(quality_scores) if quality_scores else 0.5
    
    def _calculate_diversity(self, training_data: Any) -> float:
        """Calculate diversity score for training data."""
        # Simplified diversity calculation
        # In practice, this would analyze the actual training data
        return 0.5 + self.config.diversity_bonus
    
    def _hash_contribution(
        self,
        user_id: str,
        deltas: Dict[str, torch.Tensor]
    ) -> str:
        """Create unique hash for contribution."""
        hasher = hashlib.sha256()
        hasher.update(user_id.encode())
        
        for name, delta in sorted(deltas.items()):
            hasher.update(name.encode())
            hasher.update(delta.cpu().numpy().tobytes())
        
        return hasher.hexdigest()
    
    def _group_by_layer(self) -> Dict[str, List[Tuple[str, torch.Tensor, float]]]:
        """Group contributions by layer."""
        layer_contributions = {}
        
        for contribution in self.contribution_buffer:
            user_id = contribution['user_id']
            quality = contribution['quality_score']
            
            for layer_name, delta in contribution['deltas'].items():
                if layer_name not in layer_contributions:
                    layer_contributions[layer_name] = []
                
                layer_contributions[layer_name].append((user_id, delta, quality))
        
        return layer_contributions
    
    def _byzantine_robust_aggregate(
        self,
        contributions: List[Tuple[str, torch.Tensor, float]]
    ) -> torch.Tensor:
        """Byzantine-robust aggregation using coordinate-wise median."""
        deltas = torch.stack([delta for _, delta, _ in contributions])
        
        # Compute coordinate-wise median
        median = deltas.median(dim=0)[0]
        
        # Filter out outliers (Byzantine contributions)
        distances = torch.norm(deltas - median, dim=1)
        threshold = distances.median() + 2 * distances.std()
        valid_mask = distances < threshold
        
        # Average valid contributions
        valid_deltas = deltas[valid_mask]
        if valid_deltas.numel() > 0:
            return valid_deltas.mean(dim=0)
        else:
            return median
    
    def _krum_aggregate(
        self,
        contributions: List[Tuple[str, torch.Tensor, float]]
    ) -> torch.Tensor:
        """Krum aggregation for Byzantine robustness."""
        deltas = torch.stack([delta for _, delta, _ in contributions])
        n = len(deltas)
        f = int(self.config.byzantine_threshold * n)  # Number of Byzantine clients
        
        # Compute pairwise distances
        distances = torch.zeros(n, n)
        for i in range(n):
            for j in range(i + 1, n):
                dist = torch.norm(deltas[i] - deltas[j])
                distances[i, j] = dist
                distances[j, i] = dist
        
        # Compute Krum scores
        scores = []
        for i in range(n):
            # Sort distances and sum smallest n-f-2
            sorted_dists = distances[i].sort()[0]
            score = sorted_dists[:n-f-1].sum()
            scores.append(score)
        
        # Select client with minimum score
        best_idx = torch.tensor(scores).argmin()
        return deltas[best_idx]
    
    def _trimmed_mean_aggregate(
        self,
        contributions: List[Tuple[str, torch.Tensor, float]]
    ) -> torch.Tensor:
        """Trimmed mean aggregation."""
        deltas = torch.stack([delta for _, delta, _ in contributions])
        n = len(deltas)
        
        # Number of values to trim from each end
        trim_count = int(n * self.config.trim_percent / 2)
        
        if trim_count > 0:
            # Sort along first dimension
            sorted_deltas = deltas.sort(dim=0)[0]
            # Trim and average
            trimmed = sorted_deltas[trim_count:-trim_count]
            return trimmed.mean(dim=0)
        else:
            return deltas.mean(dim=0)
    
    def _median_aggregate(
        self,
        contributions: List[Tuple[str, torch.Tensor, float]]
    ) -> torch.Tensor:
        """Median aggregation."""
        deltas = torch.stack([delta for _, delta, _ in contributions])
        return deltas.median(dim=0)[0]
    
    def _weighted_mean_aggregate(
        self,
        contributions: List[Tuple[str, torch.Tensor, float]]
    ) -> torch.Tensor:
        """Weighted mean based on contributor reputation."""
        weighted_sum = None
        total_weight = 0.0
        
        for user_id, delta, quality in contributions:
            if user_id in self.contributors:
                weight = self.contributors[user_id].get_weight() * quality
            else:
                weight = quality
            
            if weighted_sum is None:
                weighted_sum = delta * weight
            else:
                weighted_sum += delta * weight
            
            total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else weighted_sum
    
    def _mean_aggregate(
        self,
        contributions: List[Tuple[str, torch.Tensor, float]]
    ) -> torch.Tensor:
        """Simple mean aggregation."""
        deltas = torch.stack([delta for _, delta, _ in contributions])
        return deltas.mean(dim=0)
    
    def _add_differential_privacy(
        self,
        aggregated: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Add differential privacy noise to aggregated deltas."""
        noisy_aggregated = {}
        
        for layer_name, delta in aggregated.items():
            # Calculate sensitivity (L2 norm)
            sensitivity = torch.norm(delta) / len(self.contribution_buffer)
            
            # Calculate noise scale
            noise_scale = sensitivity * self.config.noise_scale / self.config.privacy_epsilon
            
            # Add Gaussian noise
            noise = torch.randn_like(delta) * noise_scale
            noisy_aggregated[layer_name] = delta + noise
        
        return noisy_aggregated
    
    def _distribute_rewards(self):
        """Distribute rewards to contributors based on quality."""
        total_quality = sum(c['quality_score'] for c in self.contribution_buffer)
        
        if total_quality == 0:
            return
        
        for contribution in self.contribution_buffer:
            user_id = contribution['user_id']
            quality = contribution['quality_score']
            
            # Calculate reward
            quality_reward = (quality / total_quality) * self.config.reward_pool * self.config.quality_weight
            participation_reward = (1.0 / len(self.contribution_buffer)) * self.config.reward_pool * self.config.participation_weight
            
            total_reward = quality_reward + participation_reward
            
            # Update contributor profile
            if user_id in self.contributors:
                self.contributors[user_id].total_rewards += total_reward
            
            print(f"Rewarded {user_id}: {total_reward:.2f} tokens")
    
    def apply_aggregated_deltas(
        self,
        model: nn.Module,
        base_model: nn.Module,
        alpha: float = 0.1
    ) -> nn.Module:
        """
        Apply aggregated deltas to a model.
        
        Args:
            model: Model to update
            base_model: Base model
            alpha: Learning rate for applying deltas
            
        Returns:
            Updated model
        """
        if not self.aggregated_deltas:
            print("No aggregated deltas available")
            return model
        
        for name, param in model.named_parameters():
            if name in self.aggregated_deltas and name in dict(base_model.named_parameters()):
                base_param = dict(base_model.named_parameters())[name]
                
                # Apply aggregated delta with learning rate
                param.data = base_param.data + alpha * self.aggregated_deltas[name]
        
        return model
    
    def get_contributor_stats(self) -> List[Dict[str, Any]]:
        """Get statistics for all contributors."""
        return [profile.to_dict() for profile in self.contributors.values()]
    
    def export_checkpoint(self, path: str):
        """Export DeltaSoup state."""
        checkpoint = {
            'config': self.config,
            'contributors': {
                uid: profile.to_dict()
                for uid, profile in self.contributors.items()
            },
            'aggregated_deltas': self.aggregated_deltas,
            'contribution_count': sum(p.contribution_count for p in self.contributors.values())
        }
        torch.save(checkpoint, path)
        print(f"DeltaSoup checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str):
        """Load DeltaSoup state."""
        checkpoint = torch.load(path)
        self.config = checkpoint['config']
        
        # Restore contributor profiles
        self.contributors = {}
        for uid, profile_dict in checkpoint['contributors'].items():
            profile = ContributorProfile(uid)
            profile.contribution_count = profile_dict['contribution_count']
            profile.reputation_score = profile_dict['reputation_score']
            profile.total_rewards = profile_dict['total_rewards']
            self.contributors[uid] = profile
        
        self.aggregated_deltas = checkpoint['aggregated_deltas']
        print(f"DeltaSoup checkpoint loaded from {path}")
        print(f"Total contributions: {checkpoint['contribution_count']}")