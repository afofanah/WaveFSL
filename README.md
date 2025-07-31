# WaveFSL

WaveFSL: Wave Interference-Based Meta-Learning for Few-Shot Cross-Modality Traffic Forecasting

## Model Versions

This repository contains two versions of the Wave Traffic Flow model:

### Version 1 (Original): `model_V2.py`
- Full-featured model with extensive wave processing capabilities
- Advanced components: SharedWaveGenerator, WaveCInterference, WASA attention
- Complex multi-scale temporal processing with adaptive blocks
- Enhanced loss function with physics-aware constraints
- Best for: Research applications requiring maximum flexibility and features

### Version 2 (Improved): `model.py` 
- Streamlined architecture optimized for performance and simplicity
- Simplified components: TrafficWaveGenerator, TrafficInterferenceProcessor
- Cleaner temporal processing with standard convolutions
- Focused loss function for better convergence
- Best for: Production deployments and faster training

## What Does This Model Do?

This model predicts future traffic flow on road networks by:
- Learning traffic patterns as waves - Traffic flow naturally has wave-like patterns (rush hours, congestion waves, etc.)
- Adapting quickly to new locations - Uses meta-learning to work on new road networks with minimal training data
- Handling multiple time scales - Captures both short-term (minutes) and long-term (hours) traffic patterns

## Key Features

### Wave-Based Traffic Modeling
- Models traffic flow as combinations of wave patterns
- Captures natural traffic rhythms like rush hours and daily cycles
- Handles wave interference between different traffic streams

### Few-Shot Learning
- Can adapt to new traffic locations with just a few days of data
- Uses Meta-Learning (MAML) to learn how to quickly adapt
- Pre-trains on multiple datasets, then fine-tunes on target location

### Multi-Scale Processing
- Analyzes traffic patterns at different time scales simultaneously
- Uses adaptive temporal blocks with multiple kernel sizes
- Combines short-term and long-term dependencies

### Attention Mechanisms
- Wave-Aware Spatial Attention (WASA) focuses on important traffic patterns
- Temporal attention for sequence modeling
- Learns which wave components are most important

## Model Architecture

### Core Pipeline (Both Versions)
```
Input Traffic Data
        ↓
Dynamic Input Projection (handles different road network sizes)
        ↓
Wave Generation (creates traffic wave patterns)
        ↓
Wave Interference Processing (models wave interactions)
        ↓
Wave-Aware Attention (focuses on important patterns)
        ↓
Multi-Scale Temporal Processing (different time scales)
        ↓
Few-Shot Adaptation (adapts to target location)
        ↓
Traffic Flow Predictions
```

### Key Architectural Differences

Version 1 (WaveFSL):
- `SharedWaveGenerator` + `TrafficWaveGenerator` for wave creation
- `WaveCInterference` with advanced traffic pattern analysis
- `WASA` (Wave-Aware Spatial Attention) with frequency band processing
- `AdaptiveTemporalBlock` with dynamic kernel selection
- `WaveAwareHead` for prediction with frequency analysis

Version 2 (AdaptiveWaveTrafficFlowModel):
- `TrafficWaveGenerator` for streamlined wave creation
- `TrafficInterferenceProcessor` with focused pattern analysis
- `TrafficWaveAttention` with simplified frequency processing
- Standard temporal convolutions with multiple kernel sizes
- Direct prediction head with layered approach



### Key Parameters
- `--test_dataset`: Target traffic dataset (e.g., 'pems-bay', 'metr-la')
- `--target_days`: Days of target data for adaptation (default: 3)
- `--wave_components`: Number of wave patterns to learn (default: 8)
- `--frequency_bands`: Frequency bands for wave analysis (default: 12)
- `--source_epochs`: Meta-training epochs (default: 300)
- `--target_epochs`: Fine-tuning epochs (default: 250)

### Configuration
Model behaviour is controlled by `config.yaml`:
- Model architecture settings
- Training hyperparameters
- Data preprocessing options
- Loss function weights

## Requirements

- Python 3.8+
- PyTorch 1.9+
- PyTorch Geometric
- NumPy, SciPy
- YAML, tqdm

## File Structure

├── main.py                    # Main training script
├── models/
│   ├── model_V2.py           # Original full-featured model (WaveFSL)
│   └── model_improved.py     # Improved streamlined model (AdaptiveWaveTrafficFlowModel)
├── train.py                  # Training and MAML implementation
├── utils.py                  # Utility functions
├── datasets.py               # Data loading and preprocessing
└── config.yaml               # Configuration file



## Citation

If you use this model in your research, please cite our work:

@article{wave_traffic_prediction,
  title={WaveFSL: Wave Interference-Based Meta-Learning for Few-Shot Cross-Modality Traffic Forecasting},
  author={AJF},
  journal={[Journal Name]},
  year={2024}
}
