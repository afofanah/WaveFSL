"""
Wave Traffic Flow Model - Improved Version
=========================================

A novel traffic flow prediction model based on wave interference principles and few-shot learning.
Enhanced with better temporal modeling and prediction accuracy optimizations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Tuple, List, Optional, Dict, Union
from einops import rearrange, repeat


class TrafficWaveGenerator(nn.Module):
    def __init__(self, input_dim: int, wave_components: int = 8):
        super().__init__()
        self.input_dim = input_dim
        self.wave_components = wave_components
        
        self.flow_amplitude_generators = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, input_dim),
                nn.LayerNorm(input_dim),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Sigmoid()
            ) for _ in range(wave_components)
        ])
        
        self.wave_numbers = nn.ParameterList([
            nn.Parameter(torch.ones(1, input_dim) * (i + 1) * 0.2)
            for i in range(wave_components)
        ])
        
        self.phase_shifts = nn.ParameterList([
            nn.Parameter(torch.randn(1, input_dim) * 0.05)
            for _ in range(wave_components)
        ])
        
        self.frequency_modulators = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, input_dim // 2),
                nn.GELU(),
                nn.Linear(input_dim // 2, input_dim),
                nn.Tanh()
            ) for _ in range(wave_components)
        ])
        
        self._initialize_weights()
        
    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        batch_size, seq_len, input_dim = x.shape
        time_coords = torch.linspace(0, 2*math.pi, seq_len, device=x.device).view(1, seq_len, 1)
        waves = []
        
        for i in range(self.wave_components):
            amplitude = self.flow_amplitude_generators[i](x)
            freq_mod = self.frequency_modulators[i](x.mean(dim=1, keepdim=True))
            effective_wave_number = self.wave_numbers[i] * (1 + 0.2 * freq_mod)
            wave_phase = effective_wave_number * time_coords + self.phase_shifts[i]
            
            if i % 4 == 0:
                wave = amplitude * torch.sin(wave_phase)
            elif i % 4 == 1:
                wave = amplitude * torch.cos(wave_phase)
            elif i % 4 == 2:
                wave = amplitude * torch.sin(wave_phase + math.pi/3)
            else:
                wave = amplitude * torch.cos(wave_phase + math.pi/6)
            
            waves.append(wave)
        
        return waves


class TrafficInterferenceProcessor(nn.Module):
    def __init__(self, input_dim: int, wave_components: int = 8):
        super().__init__()
        self.input_dim = input_dim
        self.wave_components = wave_components
        
        self.flow_harmony_analyzer = nn.Sequential(
            nn.Linear(wave_components * input_dim, input_dim * 2),
            nn.LayerNorm(input_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(input_dim * 2, input_dim),
            nn.GELU(),
            nn.Linear(input_dim, 1),
            nn.Sigmoid()
        )
        
        self.congestion_analyzer = nn.Sequential(
            nn.Linear(wave_components * input_dim, input_dim * 2),
            nn.LayerNorm(input_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(input_dim * 2, input_dim),
            nn.GELU(),
            nn.Linear(input_dim, 1),
            nn.Sigmoid()
        )
        
        self.interference_weights = nn.Parameter(
            torch.eye(wave_components) * 0.8 + torch.ones(wave_components, wave_components) * 0.2 / wave_components
        )
        
        self.pattern_enhancer = nn.Sequential(
            nn.Linear(input_dim, input_dim * 3),
            nn.LayerNorm(input_dim * 3),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(input_dim * 3, input_dim * 2),
            nn.GELU(),
            nn.Linear(input_dim * 2, input_dim),
            nn.LayerNorm(input_dim)
        )
        
        self._initialize_weights()
        
    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
    def compute_traffic_interference(self, waves: List[torch.Tensor]) -> torch.Tensor:
        wave_stack = torch.stack(waves, dim=-1)
        batch_size, seq_len, input_dim, num_waves = wave_stack.shape
        interference_sum = torch.zeros(batch_size, seq_len, input_dim, device=wave_stack.device)
        
        weights = F.softmax(self.interference_weights, dim=-1)
        
        for i in range(num_waves):
            for j in range(num_waves):
                weight = weights[i, j]
                if i == j:
                    interference = weight * wave_stack[:, :, :, i]
                else:
                    interference = weight * wave_stack[:, :, :, i] * wave_stack[:, :, :, j]
                interference_sum += interference
                
        return interference_sum / num_waves
    
    def analyze_traffic_patterns(self, waves: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        wave_concat = torch.cat([w.flatten(start_dim=-1) for w in waves], dim=-1)
        flow_harmony = self.flow_harmony_analyzer(wave_concat)
        congestion_level = self.congestion_analyzer(wave_concat)
        return flow_harmony, congestion_level
    
    def forward(self, support_waves: List[torch.Tensor], 
                query_waves: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        support_pattern = self.compute_traffic_interference(support_waves)
        query_pattern = self.compute_traffic_interference(query_waves)
        
        cross_waves = []
        for s_wave, q_wave in zip(support_waves, query_waves):
            cross_wave = 0.7 * s_wave + 0.3 * q_wave
            cross_waves.append(cross_wave)
        
        cross_pattern = self.compute_traffic_interference(cross_waves)
        flow_harmony, congestion_level = self.analyze_traffic_patterns(cross_waves)
        
        enhanced_features = self.pattern_enhancer(cross_pattern)
        final_interference = enhanced_features + cross_pattern
        
        return {
            'interference_pattern': final_interference,
            'cross_pattern': cross_pattern,
            'flow_harmony': flow_harmony,
            'congestion_level': congestion_level,
            'support_pattern': support_pattern,
            'query_pattern': query_pattern
        }


class TrafficWaveAttention(nn.Module):
    def __init__(self, input_dim: int, num_frequency_bands: int = 12):
        super().__init__()
        self.input_dim = input_dim
        self.num_frequency_bands = num_frequency_bands
        
        # Calculate band dimension to ensure proper concatenation
        self.band_dim = max(1, input_dim // num_frequency_bands)
        self.total_band_dim = self.band_dim * num_frequency_bands
        
        self.frequency_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, self.band_dim),
                nn.GELU()
            ) for _ in range(num_frequency_bands)
        ])
        
        # Project concatenated bands back to input_dim if needed
        self.band_projection = nn.Linear(self.total_band_dim, input_dim) if self.total_band_dim != input_dim else nn.Identity()
        
        self.resonance_detector = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.LayerNorm(input_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(input_dim, input_dim),
            nn.Sigmoid()
        )
        
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        self.band_combiner = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.LayerNorm(input_dim * 2),
            nn.GELU(),
            nn.Linear(input_dim * 2, input_dim)
        )
        
        self._initialize_weights()
        
    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
    def forward(self, interference_features: torch.Tensor) -> torch.Tensor:
        frequency_components = []
        for encoder in self.frequency_encoders:
            freq_component = encoder(interference_features)
            frequency_components.append(freq_component)
        
        combined_frequencies = torch.cat(frequency_components, dim=-1)
        combined_frequencies = self.band_projection(combined_frequencies)
        
        resonance_weights = self.resonance_detector(combined_frequencies)
        attended_features = interference_features * resonance_weights
        
        temporal_attended, _ = self.temporal_attention(
            attended_features, attended_features, attended_features
        )
        
        output = self.band_combiner(temporal_attended + attended_features)
        return output


class FewShotTrafficPredictor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        self.support_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
        self.query_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
        self.prototype_generator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        self.adaptation_network = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim)
        )
        
        self._initialize_weights()
        
    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
    def forward(self, support_features: torch.Tensor, 
                query_features: torch.Tensor) -> torch.Tensor:
        support_encoded = self.support_encoder(support_features)
        query_encoded = self.query_encoder(query_features)
        
        support_prototype = self.prototype_generator(support_encoded.mean(dim=1, keepdim=True))
        
        batch_size, query_len, _ = query_encoded.shape
        support_expanded = support_prototype.expand(-1, query_len, -1)
        
        combined_features = torch.cat([query_encoded, support_expanded], dim=-1)
        adapted_features = self.adaptation_network(combined_features)
        
        return adapted_features


class DynamicInputProjection(nn.Module):
    def __init__(self, target_dim: int, max_input_dim: int = 2000):
        super().__init__()
        self.target_dim = target_dim
        self.max_input_dim = max_input_dim
        
        self.projection_cache = nn.ModuleDict()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, input_dim = x.shape
        cache_key = str(input_dim)
        
        if cache_key not in self.projection_cache:
            projection = nn.Sequential(
                nn.Linear(input_dim, self.target_dim),
                nn.LayerNorm(self.target_dim),
                nn.GELU()
            )
            
            for module in projection.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
            
            self.projection_cache[cache_key] = projection.to(x.device)
        
        projected = self.projection_cache[cache_key](x)
        return projected


class WaveFSL(nn.Module):
    def __init__(self,
                 target_dim: int = 128,
                 hidden_dim: int = 128,
                 output_dim: int = 1,
                 wave_components: int = 8,
                 frequency_bands: int = 12,
                 prediction_horizon: int = 12,
                 max_input_dim: int = 2000,
                 kernel_sizes: List[int] = [3, 5, 7]):
        super().__init__()
        
        self.target_dim = target_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.prediction_horizon = prediction_horizon
        self.kernel_sizes = kernel_sizes
        
        self.input_projection = DynamicInputProjection(target_dim, max_input_dim)
        
        self.support_wave_generator = TrafficWaveGenerator(target_dim, wave_components)
        self.query_wave_generator = TrafficWaveGenerator(target_dim, wave_components)
        
        self.interference_processor = TrafficInterferenceProcessor(target_dim, wave_components)
        self.wave_attention = TrafficWaveAttention(target_dim, frequency_bands)
        self.few_shot_predictor = FewShotTrafficPredictor(target_dim, hidden_dim)
        
        self.temporal_scales = nn.ModuleList([
            self._make_temporal_block(target_dim, kernel_size)
            for kernel_size in kernel_sizes
        ])
        
        self.feature_integration = nn.Sequential(
            nn.Linear(target_dim * len(kernel_sizes), target_dim * 2),
            nn.LayerNorm(target_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(target_dim * 2, target_dim)
        )
        
        self.prediction_head = nn.Sequential(
            nn.Linear(target_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, prediction_horizon * output_dim)
        )
        
        self._initialize_weights()
    
    def _make_temporal_block(self, hidden_dim: int, kernel_size: int):
        padding = kernel_size // 2
        return nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=padding),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=padding),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU()
        )
    
    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.LayerNorm, nn.BatchNorm1d)):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, support_data: torch.Tensor, 
                query_data: torch.Tensor) -> Dict[str, torch.Tensor]:
        
        support_features = self.input_projection(support_data)
        query_features = self.input_projection(query_data)
        
        support_waves = self.support_wave_generator(support_features)
        query_waves = self.query_wave_generator(query_features)
        
        interference_results = self.interference_processor(support_waves, query_waves)
        interference_features = interference_results['interference_pattern']
        
        attended_features = self.wave_attention(interference_features)
        adapted_features = self.few_shot_predictor(support_features, attended_features)
        
        features_for_conv = adapted_features.transpose(1, 2)
        
        multiscale_features = []
        for temporal_processor in self.temporal_scales:
            ms_features = temporal_processor(features_for_conv)
            multiscale_features.append(ms_features.transpose(1, 2))
        
        integrated_features = self.feature_integration(torch.cat(multiscale_features, dim=-1))
        final_features = integrated_features + adapted_features
        
        prediction_input = final_features[:, -1, :]
        predictions = self.prediction_head(prediction_input)
        predictions = predictions.view(-1, self.prediction_horizon, self.output_dim)
        
        return {
            'predictions': predictions,
            'interference_pattern': interference_features,
            'flow_harmony': interference_results['flow_harmony'],
            'congestion_level': interference_results['congestion_level'],
            'adapted_features': adapted_features,
            'final_features': final_features,
            'multiscale_features': multiscale_features
        }


class TrafficWaveLoss(nn.Module):
    def __init__(self,
                 prediction_weight: float = 1.0,
                 harmony_weight: float = 0.1,
                 interference_weight: float = 0.05,
                 smoothness_weight: float = 0.02,
                 consistency_weight: float = 0.03):
        super().__init__()
        self.prediction_weight = prediction_weight
        self.harmony_weight = harmony_weight
        self.interference_weight = interference_weight
        self.smoothness_weight = smoothness_weight
        self.consistency_weight = consistency_weight
        
    def prediction_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        mse_loss = F.mse_loss(predictions, targets)
        mae_loss = F.l1_loss(predictions, targets)
        huber_loss = F.huber_loss(predictions, targets, delta=1.0)
        return 0.6 * mse_loss + 0.2 * mae_loss + 0.2 * huber_loss
    
    def flow_harmony_loss(self, flow_harmony: torch.Tensor) -> torch.Tensor:
        return -torch.mean(flow_harmony)
    
    def interference_coherence_loss(self, interference_pattern: torch.Tensor) -> torch.Tensor:
        temporal_diff = interference_pattern[:, 1:] - interference_pattern[:, :-1]
        temporal_smoothness = torch.mean(temporal_diff ** 2)
        feature_var = torch.var(interference_pattern, dim=-1).mean()
        return temporal_smoothness + 0.1 * feature_var
    
    def prediction_smoothness_loss(self, predictions: torch.Tensor) -> torch.Tensor:
        temporal_diff = predictions[:, 1:] - predictions[:, :-1]
        return torch.mean(temporal_diff ** 2)
    
    def multiscale_consistency_loss(self, multiscale_features: List[torch.Tensor]) -> torch.Tensor:
        if len(multiscale_features) < 2:
            return torch.tensor(0.0, device=multiscale_features[0].device)
        
        consistency_loss = 0.0
        for i in range(len(multiscale_features)):
            for j in range(i + 1, len(multiscale_features)):
                feat_i = multiscale_features[i].mean(dim=1)
                feat_j = multiscale_features[j].mean(dim=1)
                consistency_loss += F.mse_loss(feat_i, feat_j)
        
        return consistency_loss / (len(multiscale_features) * (len(multiscale_features) - 1) / 2)
    
    def forward(self, outputs: Dict[str, torch.Tensor], 
                targets: torch.Tensor) -> Dict[str, torch.Tensor]:
        predictions = outputs['predictions']
        
        pred_loss = self.prediction_loss(predictions, targets)
        harmony_loss = self.flow_harmony_loss(outputs['flow_harmony'])
        interference_loss = self.interference_coherence_loss(outputs['interference_pattern'])
        smoothness_loss = self.prediction_smoothness_loss(predictions)
        consistency_loss = self.multiscale_consistency_loss(outputs.get('multiscale_features', []))
        
        total_loss = (
            self.prediction_weight * pred_loss +
            self.harmony_weight * harmony_loss +
            self.interference_weight * interference_loss +
            self.smoothness_weight * smoothness_loss +
            self.consistency_weight * consistency_loss
        )
        
        return {
            'total_loss': total_loss,
            'prediction_loss': pred_loss,
            'harmony_loss': harmony_loss,
            'interference_loss': interference_loss,
            'smoothness_loss': smoothness_loss,
            'consistency_loss': consistency_loss
        }


def create_adaptive_model_for_args(config: dict, args) -> WaveFSL:
    task_config = config.get('task', {})
    model_config = config.get('model', {})
    
    return WaveFSL(
        target_dim=model_config.get('hidden_dim', 128),
        hidden_dim=model_config.get('hidden_dim', 128),
        output_dim=model_config.get('output_dim', 1),
        wave_components=getattr(args, 'wave_components', model_config.get('wave_components', 8)),
        frequency_bands=getattr(args, 'frequency_bands', model_config.get('frequency_bands', 12)),
        prediction_horizon=task_config.get('pred_num', 12),
        kernel_sizes=[3, 5, 7]
    )


def create_adaptive_model_from_config(config: dict) -> WaveFSL:
    task_config = config.get('task', {})
    model_config = config.get('model', {})
    
    return WaveFSL(
        target_dim=model_config.get('hidden_dim', 128),
        hidden_dim=model_config.get('hidden_dim', 128),
        output_dim=model_config.get('output_dim', 1),
        wave_components=model_config.get('wave_components', 8),
        frequency_bands=model_config.get('frequency_bands', 12),
        prediction_horizon=task_config.get('pred_num', 12),
        kernel_sizes=[3, 5, 7]
    )


def create_loss_function(prediction_weight: float = 1.0,
                        harmony_weight: float = 0.1,
                        interference_weight: float = 0.05,
                        smoothness_weight: float = 0.02,
                        consistency_weight: float = 0.03) -> TrafficWaveLoss:
    return TrafficWaveLoss(
        prediction_weight=prediction_weight,
        harmony_weight=harmony_weight,
        interference_weight=interference_weight,
        smoothness_weight=smoothness_weight,
        consistency_weight=consistency_weight
    )


WaveTrafficFlowModelForFewShot = WaveFSL