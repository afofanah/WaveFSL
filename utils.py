import os
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import Dict, List, Tuple, Optional, Union
import pickle
import json
import yaml
import time
import scipy.sparse as sp
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def asym_adj(adj: torch.Tensor) -> np.ndarray:
    adj = adj.cpu().numpy()
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat = sp.diags(d_inv)
    return d_mat.dot(adj).astype(np.float32).todense()


def get_normalized_adj(A: np.ndarray) -> np.ndarray:
    A = A + np.diag(np.ones(A.shape[0], dtype=np.float32))
    D = np.array(np.sum(A, axis=1)).reshape((-1,))
    D[D <= 1e-5] = 1e-5
    diag = np.reciprocal(np.sqrt(D))
    A_wave = np.multiply(np.multiply(diag.reshape((-1, 1)), A),
                        diag.reshape((1, -1)))
    return A_wave


def generate_dataset(X: np.ndarray, 
                    num_timesteps_input: int,
                    num_timesteps_output: int, 
                    means: np.ndarray, 
                    stds: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
    indices = [(i, i + (num_timesteps_input + num_timesteps_output)) for i
               in range(X.shape[2] - (num_timesteps_input + num_timesteps_output) + 1)]

    features, target = [], []
    for i, j in indices:
        features.append(
            X[:, :, i: i + num_timesteps_input].transpose((0, 2, 1))
        )
        target.append(X[:, 0, i + num_timesteps_input: j] * stds[0] + means[0])

    return torch.from_numpy(np.array(features)), torch.from_numpy(np.array(target))


def metric_func(pred: np.ndarray, y: np.ndarray, times: int) -> Dict[str, np.ndarray]:
    result = {
        'MSE': np.zeros(times),
        'RMSE': np.zeros(times),
        'MAE': np.zeros(times),
        'MAPE': np.zeros(times)
    }

    def cal_MAPE(pred_val: np.ndarray, y_val: np.ndarray) -> float:
        if pred_val.ndim > 1:
            pred_val = pred_val.flatten()
            y_val = y_val.flatten()
        
        mask = y_val != 0
        if np.sum(mask) == 0:
            return 0.0
        
        diff = np.abs(y_val[mask] - pred_val[mask])
        return np.mean(diff / np.abs(y_val[mask]))

    if pred.shape != y.shape:
        min_samples = min(pred.shape[0], y.shape[0])
        min_times = min(pred.shape[1] if pred.ndim > 1 else 1, 
                       y.shape[1] if y.ndim > 1 else 1, 
                       times)
        
        if pred.ndim == 3 and y.ndim == 3:
            min_nodes = min(pred.shape[2], y.shape[2])
            pred = pred[:min_samples, :min_times, :min_nodes]
            y = y[:min_samples, :min_times, :min_nodes]
        elif pred.ndim == 2 and y.ndim == 2:
            pred = pred[:min_samples, :min_times]
            y = y[:min_samples, :min_times]
        else:
            if pred.ndim == 3:
                pred = pred.reshape(pred.shape[0], pred.shape[1], -1)
            if y.ndim == 3:
                y = y.reshape(y.shape[0], y.shape[1], -1)
            
            if pred.ndim != y.ndim:
                if pred.ndim == 2 and y.ndim == 3:
                    pred = pred[:, :, np.newaxis]
                elif pred.ndim == 3 and y.ndim == 2:
                    y = y[:, :, np.newaxis]
    
    if pred.ndim == 2:
        for i in range(min(times, pred.shape[1])):
            y_i = y[:, i]
            pred_i = pred[:, i]
            
            valid_mask = np.isfinite(y_i) & np.isfinite(pred_i)
            y_i = y_i[valid_mask]
            pred_i = pred_i[valid_mask]
            
            if len(y_i) > 0:
                result['MSE'][i] = mean_squared_error(y_i, pred_i)
                result['RMSE'][i] = np.sqrt(result['MSE'][i])
                result['MAE'][i] = mean_absolute_error(y_i, pred_i)
                result['MAPE'][i] = cal_MAPE(pred_i, y_i)
    
    elif pred.ndim == 3:
        for i in range(min(times, pred.shape[1])):
            y_i = y[:, i, :].flatten()
            pred_i = pred[:, i, :].flatten()
            
            min_len = min(len(y_i), len(pred_i))
            y_i = y_i[:min_len]
            pred_i = pred_i[:min_len]
            
            valid_mask = np.isfinite(y_i) & np.isfinite(pred_i)
            y_i = y_i[valid_mask]
            pred_i = pred_i[valid_mask]
            
            if len(y_i) > 0:
                result['MSE'][i] = mean_squared_error(y_i, pred_i)
                result['RMSE'][i] = np.sqrt(result['MSE'][i])
                result['MAE'][i] = mean_absolute_error(y_i, pred_i)
                result['MAPE'][i] = cal_MAPE(pred_i, y_i)

    return result


def result_print(result: Dict[str, np.ndarray], info_name: str = 'Evaluate'):
    total_MSE = result['MSE']
    total_RMSE = result['RMSE']
    total_MAE = result['MAE']
    total_MAPE = result['MAPE']
    
    print(f"========== {info_name} results ==========")
    
    max_len = min(6, len(total_MAE))
    
    mae_values = [f"{total_MAE[i]:.3f}" for i in range(max_len)]
    mape_values = [f"{total_MAPE[i]*100:.3f}" for i in range(max_len)]
    rmse_values = [f"{total_RMSE[i]:.3f}" for i in range(max_len)]
    
    print(" MAE: " + "/ ".join(mae_values))
    print("MAPE: " + "/ ".join(mape_values))
    print("RMSE: " + "/ ".join(rmse_values))
    
    avg_mae = np.mean(total_MAE[:max_len])
    avg_mape = np.mean(total_MAPE[:max_len]) * 100
    avg_rmse = np.mean(total_RMSE[:max_len])
    
    print(f"Average - MAE: {avg_mae:.3f}, MAPE: {avg_mape:.3f}%, RMSE: {avg_rmse:.3f}")
    print("---------------------------------------")


def load_data(dataset_name: str, stage: str) -> Tuple[torch.Tensor, np.ndarray, np.ndarray, np.ndarray]:
    A = np.load(f"data/{dataset_name}/matrix.npy")
    A = get_normalized_adj(A)
    A = torch.from_numpy(A)
    
    X = np.load(f"data/{dataset_name}/dataset.npy")
    X = X.transpose((1, 2, 0))
    X = X.astype(np.float32)

    means = np.mean(X, axis=(0, 2))
    X = X - means.reshape(1, -1, 1)
    stds = np.std(X, axis=(0, 2))
    X = X / stds.reshape(1, -1, 1)

    if stage == 'train':
        X = X[:, :, :int(X.shape[2]*0.7)]
    elif stage == 'validation':
        X = X[:, :, int(X.shape[2]*0.7):int(X.shape[2]*0.8)]
    elif stage == 'test':
        X = X[:, :, int(X.shape[2]*0.8):]
    elif stage == 'source':
        X = X
    elif stage == 'target_1day':
        X = X[:, :, :288]
    elif stage == 'target_3day':
        X = X[:, :, :288*3]
    elif stage == 'target_1week':
        X = X[:, :, :288*7]

    return A, X, means, stds


def load_config(config_path: str = 'config.yaml') -> Dict:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def save_config(config: Dict, filepath: str):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    if filepath.endswith('.yaml') or filepath.endswith('.yml'):
        with open(filepath, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
    elif filepath.endswith('.json'):
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)
    else:
        with open(filepath, 'wb') as f:
            pickle.dump(config, f)


def set_random_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_experiment_dir(base_dir: str, experiment_name: str) -> str:
    import datetime
    
    # Ensure base directory exists
    base_dir = os.path.abspath(base_dir)  # Convert to absolute path
    os.makedirs(base_dir, exist_ok=True)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(base_dir, f"{experiment_name}_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)
    
    # Create subdirectories for organized results
    subdirs = ['plots', 'checkpoints', 'logs', 'models']
    for subdir in subdirs:
        os.makedirs(os.path.join(exp_dir, subdir), exist_ok=True)
    
    print(f"Created experiment directory: {exp_dir}")
    return exp_dir


def log_metrics(metrics: Dict, epoch: int, stage: str, 
               log_file: str = None, verbose: bool = True):
    log_message = f"Epoch {epoch} [{stage}] - "
    log_message += ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
    
    if verbose:
        print(log_message)
    
    if log_file:
        with open(log_file, 'a') as f:
            f.write(log_message + "\n")


def calculate_model_size(model: torch.nn.Module) -> Dict[str, float]:
    param_size = 0
    param_sum = 0
    
    for param in model.parameters():
        param_sum += param.nelement()
        param_size += param.nelement() * param.element_size()
    
    buffer_size = 0
    buffer_sum = 0
    
    for buffer in model.buffers():
        buffer_sum += buffer.nelement()
        buffer_size += buffer.nelement() * buffer.element_size()
    
    all_size = (param_size + buffer_size) / 1024 / 1024
    
    return {
        'param_count': param_sum,
        'buffer_count': buffer_sum,
        'total_count': param_sum + buffer_sum,
        'param_size_mb': param_size / 1024 / 1024,
        'buffer_size_mb': buffer_size / 1024 / 1024,
        'total_size_mb': all_size
    }


def prepare_geometric_data(x_data: torch.Tensor, y_data: torch.Tensor, 
                          edge_index: torch.Tensor, node_num: int) -> Dict:
    batch_size, nodes, time_steps, features = x_data.shape
    
    support_data = x_data.permute(0, 2, 1, 3).contiguous()
    support_data = support_data.view(batch_size, time_steps, nodes * features)
    
    if y_data.dim() == 2:
        y_data = y_data.unsqueeze(-1)
    
    query_data = y_data.permute(0, 2, 1).contiguous()
    
    return {
        'support_data': support_data,
        'query_data': query_data,
        'edge_index': edge_index,
        'node_num': node_num,
        'original_shape': x_data.shape
    }


def convert_predictions_to_geometric(predictions: torch.Tensor, 
                                   original_shape: Tuple[int, ...]) -> torch.Tensor:
    batch_size, pred_steps, nodes = predictions.shape
    geometric_predictions = predictions.permute(0, 2, 1).contiguous()
    return geometric_predictions


def validate_config(config: Dict) -> bool:
    required_keys = ['data', 'task', 'model']
    
    for key in required_keys:
        if key not in config:
            return False
    
    if 'data_keys' not in config['data']:
        return False
    
    required_task_keys = ['his_num', 'pred_num', 'batch_size', 'task_num']
    for key in required_task_keys:
        if key not in config['task']:
            return False
    
    required_model_keys = ['update_lr', 'meta_lr', 'hidden_dim']
    for key in required_model_keys:
        if key not in config['model']:
            return False
    
    return True


def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, 
                   epoch: int, loss: float, filepath: str):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'timestamp': time.time()
    }
    torch.save(checkpoint, filepath)


def load_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, 
                   filepath: str) -> Tuple[int, float]:
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint file not found: {filepath}")
    
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    
    return epoch, loss


def get_device(device_str: str = 'auto') -> torch.device:
    if device_str == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_str)
    
    return device


# ========================= PLOTTING FUNCTIONS =========================

def plot_training_curves(train_losses: List[float], val_losses: List[float] = None, 
                         save_path: str = None, title: str = "Training Curves"):
    
    if not train_losses or len(train_losses) == 0:
        print("Warning: No training losses to plot")
        return
        
    plt.figure(figsize=(12, 8))
    
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2, marker='o', markersize=3)
    
    if val_losses and len(val_losses) > 0:
        val_epochs = range(1, len(val_losses) + 1)
        plt.plot(val_epochs, val_losses, 'r-', label='Validation Loss', linewidth=2, marker='s', markersize=3)
    
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.title(f'{title}\n(Final Loss: {train_losses[-1]:.6f})', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Add some statistics
    min_loss = min(train_losses)
    min_epoch = train_losses.index(min_loss) + 1
    plt.axhline(y=min_loss, color='green', linestyle='--', alpha=0.7, label=f'Min Loss: {min_loss:.6f} @Epoch {min_epoch}')
    plt.legend(fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training curves saved to: {save_path}")
    plt.show()


def plot_predictions_vs_actual(predictions: np.ndarray, actuals: np.ndarray, 
                              save_path: str = None, num_samples: int = 3, 
                              num_nodes: int = 3):
    
    print(f"Original shapes - Predictions: {predictions.shape}, Actuals: {actuals.shape}")
    
    # Ensure we have data to plot
    if predictions.size == 0 or actuals.size == 0:
        print("Warning: Empty data arrays, skipping prediction plot")
        return
    
    # Handle shape mismatches and ensure we have plottable data
    if predictions.shape != actuals.shape:
        print(f"Shape mismatch detected, aligning shapes...")
        
        if predictions.ndim == 2 and actuals.ndim == 3:
            # predictions: (samples, time), actuals: (samples, time, nodes)
            # Take the first node or average across nodes
            if actuals.shape[2] > 1:
                actuals = actuals[:, :, 0]  # Take first node for simplicity
                print(f"Using first node from actuals, new shape: {actuals.shape}")
            else:
                actuals = actuals.squeeze(-1)
                
        elif predictions.ndim == 3 and actuals.ndim == 2:
            # predictions: (samples, time, nodes), actuals: (samples, time)
            if predictions.shape[2] > 1:
                predictions = predictions[:, :, 0]  # Take first node
                print(f"Using first node from predictions, new shape: {predictions.shape}")
            else:
                predictions = predictions.squeeze(-1)
                
        elif predictions.ndim == 3 and actuals.ndim == 3:
            # Both 3D, align dimensions
            min_samples = min(predictions.shape[0], actuals.shape[0])
            min_time = min(predictions.shape[1], actuals.shape[1])
            min_nodes = min(predictions.shape[2], actuals.shape[2])
            predictions = predictions[:min_samples, :min_time, :min_nodes]
            actuals = actuals[:min_samples, :min_time, :min_nodes]
            print(f"Aligned 3D shapes to: {predictions.shape}")
            
        elif predictions.ndim == 2 and actuals.ndim == 2:
            # Both 2D, align dimensions
            min_samples = min(predictions.shape[0], actuals.shape[0])
            min_features = min(predictions.shape[1], actuals.shape[1])
            predictions = predictions[:min_samples, :min_features]
            actuals = actuals[:min_samples, :min_features]
            print(f"Aligned 2D shapes to: {predictions.shape}")
    
    # Determine actual plotting parameters
    actual_samples = min(num_samples, predictions.shape[0])
    
    if predictions.ndim == 3:
        actual_nodes = min(num_nodes, predictions.shape[2])
        plot_3d = True
    else:
        actual_nodes = 1
        plot_3d = False
    
    print(f"Plotting {actual_samples} samples, {actual_nodes} nodes, 3D: {plot_3d}")
    
    # Create figure
    if plot_3d and actual_nodes > 1:
        fig, axes = plt.subplots(actual_samples, actual_nodes, figsize=(5*actual_nodes, 4*actual_samples))
        if actual_samples == 1:
            axes = axes.reshape(1, -1) if actual_nodes > 1 else [[axes]]
        elif actual_nodes == 1:
            axes = [[ax] for ax in axes]
    else:
        fig, axes = plt.subplots(actual_samples, 1, figsize=(10, 4*actual_samples))
        if actual_samples == 1:
            axes = [axes]
        axes = [[ax] for ax in axes]  # Make it consistent 2D indexing
    
    # Plot data
    for i in range(actual_samples):
        if plot_3d and actual_nodes > 1:
            for j in range(actual_nodes):
                ax = axes[i][j]
                
                pred_vals = predictions[i, :, j]
                actual_vals = actuals[i, :, j]
                
                time_steps = np.arange(len(pred_vals))
                
                ax.plot(time_steps, actual_vals, 'b-', label='Actual', linewidth=2, marker='o', markersize=4)
                ax.plot(time_steps, pred_vals, 'r--', label='Predicted', linewidth=2, marker='s', markersize=4)
                
                ax.set_title(f'Sample {i+1}, Node {j+1}', fontsize=12, fontweight='bold')
                ax.set_xlabel('Time Step')
                ax.set_ylabel('Value')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                # Add value range info
                ax.text(0.02, 0.98, f'Range: [{np.min(actual_vals):.2f}, {np.max(actual_vals):.2f}]', 
                       transform=ax.transAxes, fontsize=8, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        else:
            # 2D plotting or single node
            ax = axes[i][0]
            
            if plot_3d:
                pred_vals = predictions[i, :, 0]  # Take first node
                actual_vals = actuals[i, :, 0]
            else:
                pred_vals = predictions[i, :]
                actual_vals = actuals[i, :]
            
            time_steps = np.arange(len(pred_vals))
            
            ax.plot(time_steps, actual_vals, 'b-', label='Actual', linewidth=2, marker='o', markersize=4)
            ax.plot(time_steps, pred_vals, 'r--', label='Predicted', linewidth=2, marker='s', markersize=4)
            
            ax.set_title(f'Sample {i+1}', fontsize=12, fontweight='bold')
            ax.set_xlabel('Time Step')
            ax.set_ylabel('Value')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Add value range and error info
            mae = np.mean(np.abs(pred_vals - actual_vals))
            ax.text(0.02, 0.98, f'Range: [{np.min(actual_vals):.2f}, {np.max(actual_vals):.2f}]\nMAE: {mae:.3f}', 
                   transform=ax.transAxes, fontsize=8, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Prediction plot saved to: {save_path}")
    plt.show()


def plot_error_distribution(predictions: np.ndarray, actuals: np.ndarray, 
                           save_path: str = None):
    # Handle shape mismatches before computing errors
    orig_pred_shape = predictions.shape
    orig_actual_shape = actuals.shape
    
    # Align shapes first
    if predictions.shape != actuals.shape:
        if predictions.ndim == 2 and actuals.ndim == 2:
            # Both 2D but different shapes
            min_samples = min(predictions.shape[0], actuals.shape[0])
            min_time_or_nodes = min(predictions.shape[1], actuals.shape[1])
            predictions = predictions[:min_samples, :min_time_or_nodes]
            actuals = actuals[:min_samples, :min_time_or_nodes]
        elif predictions.ndim == 3 and actuals.ndim == 3:
            # Both 3D, align all dimensions
            min_samples = min(predictions.shape[0], actuals.shape[0])
            min_time = min(predictions.shape[1], actuals.shape[1])
            min_nodes = min(predictions.shape[2], actuals.shape[2])
            predictions = predictions[:min_samples, :min_time, :min_nodes]
            actuals = actuals[:min_samples, :min_time, :min_nodes]
        elif predictions.ndim == 2 and actuals.ndim == 3:
            # predictions 2D, actuals 3D - take mean over nodes or first node
            if actuals.shape[2] > 1:
                actuals = np.mean(actuals, axis=2)  # Average over nodes
            else:
                actuals = actuals[:, :, 0]  # Take first node
            min_samples = min(predictions.shape[0], actuals.shape[0])
            min_time = min(predictions.shape[1], actuals.shape[1])
            predictions = predictions[:min_samples, :min_time]
            actuals = actuals[:min_samples, :min_time]
        elif predictions.ndim == 3 and actuals.ndim == 2:
            # predictions 3D, actuals 2D - take mean over nodes or first node
            if predictions.shape[2] > 1:
                predictions = np.mean(predictions, axis=2)  # Average over nodes
            else:
                predictions = predictions[:, :, 0]  # Take first node
            min_samples = min(predictions.shape[0], actuals.shape[0])
            min_time = min(predictions.shape[1], actuals.shape[1])
            predictions = predictions[:min_samples, :min_time]
            actuals = actuals[:min_samples, :min_time]
        else:
            # Last resort: truncate flattened arrays to same length
            pred_flat = predictions.flatten()
            actual_flat = actuals.flatten()
            min_len = min(len(pred_flat), len(actual_flat))
            predictions = pred_flat[:min_len].reshape(-1, 1)
            actuals = actual_flat[:min_len].reshape(-1, 1)
    
    # Now flatten after shape alignment
    pred_flat = predictions.flatten()
    actual_flat = actuals.flatten()
    
    # Final safety check
    if len(pred_flat) != len(actual_flat):
        min_len = min(len(pred_flat), len(actual_flat))
        pred_flat = pred_flat[:min_len]
        actual_flat = actual_flat[:min_len]
    
    # Compute errors
    errors = pred_flat - actual_flat
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Error histogram
    axes[0].hist(errors, bins=50, density=True, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0].set_title(f'Error Distribution\n(Pred: {orig_pred_shape}, Actual: {orig_actual_shape})', 
                     fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Error')
    axes[0].set_ylabel('Density')
    axes[0].grid(True, alpha=0.3)
    
    # Add statistics to the plot
    mean_error = np.mean(errors)
    std_error = np.std(errors)
    axes[0].axvline(mean_error, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_error:.4f}')
    axes[0].axvline(mean_error + std_error, color='orange', linestyle='--', alpha=0.7, label=f'±1σ: {std_error:.4f}')
    axes[0].axvline(mean_error - std_error, color='orange', linestyle='--', alpha=0.7)
    axes[0].legend()
    
    # Scatter plot
    sample_size = min(10000, len(pred_flat))  # Limit points for performance
    indices = np.random.choice(len(pred_flat), sample_size, replace=False)
    
    axes[1].scatter(actual_flat[indices], pred_flat[indices], alpha=0.5, s=10)
    min_val = min(actual_flat.min(), pred_flat.min())
    max_val = max(actual_flat.max(), pred_flat.max())
    axes[1].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
    axes[1].set_title('Predictions vs Actuals', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Actual Values')
    axes[1].set_ylabel('Predicted Values')
    axes[1].grid(True, alpha=0.3)
    
    # Calculate R²
    correlation_matrix = np.corrcoef(actual_flat[indices], pred_flat[indices])
    r_squared = correlation_matrix[0, 1] ** 2
    axes[1].text(0.05, 0.95, f'R² = {r_squared:.4f}', transform=axes[1].transAxes, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Absolute error boxplot
    residuals = np.abs(errors)
    axes[2].boxplot(residuals, vert=True)
    axes[2].set_title('Absolute Error Distribution', fontsize=14, fontweight='bold')
    axes[2].set_ylabel('Absolute Error')
    axes[2].grid(True, alpha=0.3)
    
    # Add summary statistics
    mae = np.mean(residuals)
    rmse = np.sqrt(np.mean(errors**2))
    axes[2].text(0.05, 0.95, f'MAE: {mae:.4f}\nRMSE: {rmse:.4f}', 
                transform=axes[2].transAxes,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_metrics_comparison(metrics_dict: Dict[str, Dict[str, np.ndarray]], 
                           save_path: str = None):
    
    if not metrics_dict or len(metrics_dict) == 0:
        print("Warning: No metrics to plot")
        return
    
    # Filter valid metrics
    valid_metrics = {}
    for model_name, model_metrics in metrics_dict.items():
        if model_metrics and isinstance(model_metrics, dict):
            valid_metrics[model_name] = model_metrics
    
    if len(valid_metrics) == 0:
        print("Warning: No valid metrics found")
        return
    
    metrics = ['MAE', 'RMSE', 'MAPE']
    available_metrics = []
    
    # Check which metrics are actually available
    for metric in metrics:
        if any(metric in model_metrics for model_metrics in valid_metrics.values()):
            available_metrics.append(metric)
    
    if len(available_metrics) == 0:
        print("Warning: No standard metrics (MAE, RMSE, MAPE) found in data")
        return
    
    n_metrics = len(available_metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(6*n_metrics, 6))
    
    if n_metrics == 1:
        axes = [axes]
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(valid_metrics)))
    
    print(f"Plotting {n_metrics} metrics for {len(valid_metrics)} models")
    
    for idx, metric in enumerate(available_metrics):
        ax = axes[idx]
        
        for i, (model_name, model_metrics) in enumerate(valid_metrics.items()):
            if metric in model_metrics:
                metric_values = model_metrics[metric]
                
                # Ensure we have array data
                if isinstance(metric_values, (list, np.ndarray)) and len(metric_values) > 0:
                    time_steps = range(1, len(metric_values) + 1)
                    values = metric_values * 100 if metric == 'MAPE' else metric_values
                    
                    ax.plot(time_steps, values, marker='o', label=model_name, 
                           color=colors[i], linewidth=2, markersize=6)
                    
                    # Add value annotations for key points
                    if len(values) > 0:
                        ax.annotate(f'{values[0]:.3f}', (1, values[0]), 
                                  textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)
                        if len(values) > 1:
                            ax.annotate(f'{values[-1]:.3f}', (len(values), values[-1]), 
                                      textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)
        
        ax.set_title(f'{metric} Comparison', fontsize=14, fontweight='bold')
        ax.set_xlabel('Prediction Step')
        ylabel = f'{metric} (%)' if metric == 'MAPE' else metric
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Set reasonable y-limits if possible
        if len(ax.get_lines()) > 0:
            all_y_data = []
            for line in ax.get_lines():
                all_y_data.extend(line.get_ydata())
            if all_y_data:
                y_min, y_max = min(all_y_data), max(all_y_data)
                y_range = y_max - y_min
                ax.set_ylim(y_min - 0.1*y_range, y_max + 0.1*y_range)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Metrics comparison plot saved to: {save_path}")
    plt.show()


def plot_attention_weights(attention_weights: torch.Tensor, save_path: str = None, 
                          max_nodes: int = 20):
    if attention_weights.dim() == 4:
        attention_weights = attention_weights[0, 0]
    elif attention_weights.dim() == 3:
        attention_weights = attention_weights[0]
    
    attention_weights = attention_weights[:max_nodes, :max_nodes]
    attention_np = attention_weights.detach().cpu().numpy()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(attention_np, annot=False, cmap='Blues', cbar=True, 
                square=True, linewidths=0.5)
    plt.title('Attention Weights Visualization', fontsize=16, fontweight='bold')
    plt.xlabel('Key Nodes', fontsize=14)
    plt.ylabel('Query Nodes', fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_wave_interference_pattern(interference_pattern: torch.Tensor, 
                                  save_path: str = None, max_features: int = 8):
    
    if interference_pattern is None:
        print("Warning: No interference pattern to plot")
        return
        
    if interference_pattern.dim() == 3:
        pattern = interference_pattern[0]  # Take first batch
    else:
        pattern = interference_pattern
    
    if pattern.numel() == 0:
        print("Warning: Empty interference pattern")
        return
    
    # Convert to numpy
    pattern_np = pattern.detach().cpu().numpy()
    
    # Limit features for visualization
    actual_features = min(max_features, pattern_np.shape[1])
    pattern_np = pattern_np[:, :actual_features]
    
    print(f"Plotting wave interference pattern: {pattern_np.shape}")
    
    plt.figure(figsize=(12, 8))
    
    colors = plt.cm.tab10(np.linspace(0, 1, actual_features))
    
    for i in range(actual_features):
        feature_data = pattern_np[:, i]
        plt.plot(feature_data, label=f'Feature {i+1}', linewidth=2, 
                color=colors[i], marker='o', markersize=2, alpha=0.8)
    
    plt.title(f'Wave Interference Pattern\n({actual_features} features, {pattern_np.shape[0]} time steps)', 
              fontsize=16, fontweight='bold')
    plt.xlabel('Time Steps', fontsize=14)
    plt.ylabel('Interference Amplitude', fontsize=14)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Add statistics
    mean_amplitude = np.mean(np.abs(pattern_np))
    max_amplitude = np.max(np.abs(pattern_np))
    plt.text(0.02, 0.98, f'Mean Amplitude: {mean_amplitude:.4f}\nMax Amplitude: {max_amplitude:.4f}', 
             transform=plt.gca().transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Wave interference plot saved to: {save_path}")
    plt.show()


def plot_multiscale_features(multiscale_features: List[torch.Tensor], 
                            save_path: str = None, kernel_sizes: List[int] = [3, 5, 7]):
    
    if not multiscale_features or len(multiscale_features) == 0:
        print("Warning: No multiscale features to plot")
        return
    
    # Filter out None values
    valid_features = [f for f in multiscale_features if f is not None and f.numel() > 0]
    
    if len(valid_features) == 0:
        print("Warning: No valid multiscale features found")
        return
    
    n_scales = len(valid_features)
    fig, axes = plt.subplots(n_scales, 1, figsize=(14, 4*n_scales))
    
    if n_scales == 1:
        axes = [axes]
    
    print(f"Plotting {n_scales} multiscale features")
    
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
    
    for i, features in enumerate(valid_features):
        if features.dim() == 3:
            features = features[0]  # Take first batch
        
        features_np = features.detach().cpu().numpy()
        
        # Compute statistics across feature dimension
        feature_mean = np.mean(features_np, axis=1)
        feature_std = np.std(features_np, axis=1)
        feature_max = np.max(features_np, axis=1)
        feature_min = np.min(features_np, axis=1)
        
        time_steps = np.arange(len(feature_mean))
        
        # Plot mean with error bars
        axes[i].plot(time_steps, feature_mean, linewidth=2, color=colors[i % len(colors)], 
                    marker='o', markersize=3, label='Mean')
        axes[i].fill_between(time_steps, feature_mean - feature_std, feature_mean + feature_std, 
                           alpha=0.3, color=colors[i % len(colors)], label='±1σ')
        
        # Add min/max envelope
        axes[i].plot(time_steps, feature_max, '--', alpha=0.7, color=colors[i % len(colors)], 
                    linewidth=1, label='Max')
        axes[i].plot(time_steps, feature_min, '--', alpha=0.7, color=colors[i % len(colors)], 
                    linewidth=1, label='Min')
        
        kernel_size = kernel_sizes[i] if i < len(kernel_sizes) else f"Scale {i+1}"
        axes[i].set_title(f'Scale {i+1} (Kernel Size: {kernel_size})\nShape: {features_np.shape}', 
                         fontsize=14, fontweight='bold')
        axes[i].set_xlabel('Time Steps')
        axes[i].set_ylabel('Feature Magnitude')
        axes[i].legend(loc='upper right', fontsize=10)
        axes[i].grid(True, alpha=0.3)
        
        # Add statistics
        mean_magnitude = np.mean(np.abs(feature_mean))
        axes[i].text(0.02, 0.98, f'Avg Magnitude: {mean_magnitude:.4f}', 
                    transform=axes[i].transAxes, fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Multiscale features plot saved to: {save_path}")
    plt.show()


def plot_loss_components(loss_history: Dict[str, List[float]], save_path: str = None):
    plt.figure(figsize=(15, 10))
    
    n_components = len(loss_history)
    n_cols = 3
    n_rows = (n_components + n_cols - 1) // n_cols
    
    colors = plt.cm.Set1(np.linspace(0, 1, n_components))
    
    for idx, (loss_name, loss_values) in enumerate(loss_history.items()):
        plt.subplot(n_rows, n_cols, idx + 1)
        epochs = range(1, len(loss_values) + 1)
        plt.plot(epochs, loss_values, color=colors[idx], linewidth=2)
        plt.title(f'{loss_name.replace("_", " ").title()}', fontsize=12, fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_dataset_statistics(dataset_stats: Dict, save_path: str = None):
    datasets = list(dataset_stats.keys())
    node_counts = [stats.get('node_num', 0) for stats in dataset_stats.values()]
    time_steps = [stats.get('time_step', 0) for stats in dataset_stats.values()]
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    axes[0].bar(datasets, node_counts, color='skyblue', edgecolor='black')
    axes[0].set_title('Number of Nodes per Dataset', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Node Count')
    axes[0].tick_params(axis='x', rotation=45)
    
    axes[1].bar(datasets, time_steps, color='lightcoral', edgecolor='black')
    axes[1].set_title('Time Steps per Dataset', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Time Steps')
    axes[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def save_all_plots(exp_dir: str, model_outputs: Dict, metrics: Dict, 
                  train_losses: List[float], **kwargs):
    plots_dir = os.path.join(exp_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    print(f"Generating plots in: {plots_dir}")
    
    # Plot training curves with validation
    if train_losses and len(train_losses) > 0:
        print(f"Plotting training curves with {len(train_losses)} data points")
        plot_training_curves(train_losses, 
                            save_path=os.path.join(plots_dir, 'training_curves.png'))
    else:
        print("Warning: No training losses to plot")
    
    # Plot predictions vs actual with comprehensive error handling
    if 'predictions' in kwargs and 'targets' in kwargs:
        predictions = kwargs['predictions']
        targets = kwargs['targets']
        
        print(f"Attempting to plot predictions vs actuals...")
        print(f"Predictions type: {type(predictions)}, shape: {getattr(predictions, 'shape', 'No shape')}")
        print(f"Targets type: {type(targets)}, shape: {getattr(targets, 'shape', 'No shape')}")
        
        if predictions is not None and targets is not None:
            if hasattr(predictions, 'shape') and hasattr(targets, 'shape'):
                if predictions.size > 0 and targets.size > 0:
                    plot_predictions_vs_actual(predictions, targets,
                                              save_path=os.path.join(plots_dir, 'predictions_vs_actual.png'))
                    
                    plot_error_distribution(predictions, targets,
                                           save_path=os.path.join(plots_dir, 'error_distribution.png'))
                else:
                    print("Warning: Empty prediction or target arrays")
            else:
                print("Warning: Predictions or targets don't have shape attribute")
        else:
            print("Warning: Predictions or targets are None")
    else:
        print("Warning: No predictions or targets provided")
    
    # Plot wave interference pattern with validation
    if model_outputs and 'interference_pattern' in model_outputs:
        interference_pattern = model_outputs['interference_pattern']
        if interference_pattern is not None:
            print(f"Plotting wave interference pattern: {interference_pattern.shape if hasattr(interference_pattern, 'shape') else 'No shape'}")
            plot_wave_interference_pattern(interference_pattern,
                                          save_path=os.path.join(plots_dir, 'wave_interference.png'))
        else:
            print("Warning: Interference pattern is None")
    else:
        print("Warning: No interference pattern in model outputs")
    
    # Plot multiscale features with validation
    if model_outputs and 'multiscale_features' in model_outputs:
        multiscale_features = model_outputs['multiscale_features']
        if multiscale_features is not None and len(multiscale_features) > 0:
            print(f"Plotting multiscale features: {len(multiscale_features)} scales")
            plot_multiscale_features(multiscale_features,
                                    save_path=os.path.join(plots_dir, 'multiscale_features.png'))
        else:
            print("Warning: No multiscale features or empty list")
    else:
        print("Warning: No multiscale features in model outputs")
    
    # Plot metrics comparison with validation
    if metrics and isinstance(metrics, dict):
        if any(isinstance(v, dict) for v in metrics.values()):
            print(f"Plotting metrics comparison for {len(metrics)} models/experiments")
            plot_metrics_comparison(metrics, 
                                   save_path=os.path.join(plots_dir, 'metrics_comparison.png'))
        else:
            print("Warning: Metrics dict doesn't contain nested dictionaries")
    else:
        print("Warning: No valid metrics dictionary provided")
    
    print(f"Plot generation completed. Check {plots_dir} for all plots.")


class EarlyStopping:
    def __init__(self, patience: int = 20, min_delta: float = 0.001, 
                 restore_best_weights: bool = True, min_epochs: int = 0):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.min_epochs = min_epochs
        self.best_loss = float('inf')
        self.counter = 0
        self.best_weights = None
        self.current_epoch = 0
        
    def __call__(self, val_loss: float, model: torch.nn.Module, epoch: int = None) -> bool:
        if epoch is not None:
            self.current_epoch = epoch
        else:
            self.current_epoch += 1
            
        # Don't allow early stopping before minimum epochs
        if self.current_epoch < self.min_epochs:
            if val_loss < self.best_loss - self.min_delta:
                self.best_loss = val_loss
                if self.restore_best_weights:
                    self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            return False
            
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights and self.best_weights:
                model.load_state_dict({k: v.to(next(model.parameters()).device) 
                                     for k, v in self.best_weights.items()})
            return True
        return False