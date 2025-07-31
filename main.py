import os
import argparse
import yaml
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import DataLoader
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Tuple, Any

from models.model_V2 import create_adaptive_model_for_args, create_loss_function
from train import WaveTrafficMAML,  save_maml_checkpoint
from utils import (
    load_config, save_config, set_random_seed, create_experiment_dir,
    log_metrics, get_device, validate_config, count_parameters,
    result_print, save_all_plots, EarlyStopping
)
from datasets import WaveTrafficDataset


def parse_arguments():
    parser = argparse.ArgumentParser(description='Wave Traffic Flow Prediction with MAML')
    
    parser.add_argument('--config_filename', default='config.yaml', type=str)
    parser.add_argument('--test_dataset', default='pems-bay', type=str)
    parser.add_argument('--target_days', default=3, type=int)
    
    parser.add_argument('--source_epochs', default=300, type=int)
    parser.add_argument('--target_epochs', default=250, type=int)
    parser.add_argument('--source_lr', default=1e-2, type=float)
    parser.add_argument('--target_lr', default=1e-2, type=float)

    parser.add_argument('--batch_size', default=5, type=int)
    parser.add_argument('--meta_dim', default=256, type=int) 
    parser.add_argument('--model', default='WaveTrafficFlow', type=str)
    parser.add_argument('--wave_components', default=8, type=int)
    parser.add_argument('--frequency_bands', default=12, type=int)
    
    parser.add_argument('--update_lr', default=0.01, type=float)
    parser.add_argument('--meta_lr', default=0.001, type=float)
    parser.add_argument('--loss_lambda', default=1.5, type=float)
    
    parser.add_argument('--learning_strategy', default='FewShot', type=str)
    parser.add_argument('--memo', default='wave_traffic_experiment', type=str)
    parser.add_argument('--save_dir', default='./Conference QLD/results', type=str)
    parser.add_argument('--device', default='auto', type=str)
    parser.add_argument('--seed', default=7, type=int)
    
    parser.add_argument('--log_interval', default=20, type=int)
    parser.add_argument('--save_interval', default=50, type=int)
    parser.add_argument('--eval_interval', default=25, type=int)
    
    parser.add_argument('--use_scheduler', default=True, type=bool)
    parser.add_argument('--gradient_clip', default=1.0, type=float)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    
    return parser.parse_args()


def setup_experiment(args) -> Tuple[Dict, str, torch.device]:
    set_random_seed(args.seed)
    device = get_device(args.device)
    config = load_config(args.config_filename)
    
    if not validate_config(config):
        raise ValueError("Invalid configuration file")
    
    config['model']['update_lr'] = args.update_lr
    config['model']['meta_lr'] = args.meta_lr
    config['model']['loss_lambda'] = args.loss_lambda
    config['model']['wave_components'] = args.wave_components
    config['model']['frequency_bands'] = args.frequency_bands
    config['task']['batch_size'] = args.batch_size
    
    exp_dir = create_experiment_dir(args.save_dir, f"{args.memo}_{args.test_dataset}")
    
    save_config(config, os.path.join(exp_dir, 'config.yaml'))
    save_config(vars(args), os.path.join(exp_dir, 'args.yaml'))
    
    print(f"Experiment directory: {exp_dir}")
    print(f"Device: {device}")
    print(f"Test dataset: {args.test_dataset}")
    
    return config, exp_dir, device


def create_datasets(config: Dict, args, device: torch.device) -> Tuple[WaveTrafficDataset, Any, Any]:
    print("Creating datasets...")
    
    source_dataset = WaveTrafficDataset(
        data_args=config['data'],
        task_args=config['task'],
        stage='source',
        test_data=args.test_dataset,
        add_target=True,
        target_days=args.target_days
    )
    
    target_dataset = WaveTrafficDataset(
        data_args=config['data'],
        task_args=config['task'],
        stage='target',
        test_data=args.test_dataset,
        target_days=args.target_days
    )
    
    test_dataset = WaveTrafficDataset(
        data_args=config['data'],
        task_args=config['task'],
        stage='test',
        test_data=args.test_dataset
    )
    
    target_dataloader = DataLoader(
        target_dataset,
        batch_size=config['task']['batch_size'],
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config['task']['test_batch_size'],
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    print(f"Source datasets: {source_dataset.data_list}")
    print(f"Target dataset samples: {len(target_dataset)}")
    print(f"Test dataset samples: {len(test_dataset)}")
    
    return source_dataset, target_dataloader, test_dataloader


def create_model_and_optimizer(config: Dict, args, device: torch.device) -> WaveTrafficMAML:
    print("Creating model...")
    
    base_model = create_adaptive_model_for_args(config, args)
    
    model_args = config['model'].copy()
    model_args.update({
        'update_lr': args.update_lr,
        'meta_lr': args.meta_lr,
        'loss_lambda': args.loss_lambda,
        'wave_components': args.wave_components,
        'frequency_bands': args.frequency_bands,
        'prediction_weight': config['model'].get('prediction_weight', 1.0),
        'harmony_weight': config['model'].get('harmony_weight', 0.1),
        'interference_weight': config['model'].get('interference_weight', 0.05),
        'smoothness_weight': config['model'].get('smoothness_weight', 0.02),
        'consistency_weight': config['model'].get('consistency_weight', 0.03)
    })
    
    maml_model = WaveTrafficMAML(
        model=base_model,
        model_args=model_args,
        task_args=config['task'],
        learning_strategy=args.learning_strategy
    ).to(device)
    
    total_params = count_parameters(maml_model)
    print(f"Total trainable parameters: {total_params:,}")
    
    return maml_model


def meta_train_phase(model: WaveTrafficMAML, source_dataset: WaveTrafficDataset, 
                    config: Dict, args, exp_dir: str, device: torch.device):
    print("\n" + "="*60)
    print("STARTING META-TRAINING PHASE")
    print("="*60)
    
    log_file = os.path.join(exp_dir, 'logs', 'meta_training.log')
    best_meta_loss = float('inf')
    training_losses = []
    
    early_stopping = EarlyStopping(patience=100, min_delta=0.00001, min_epochs=args.source_epochs)
    
    progress_bar = tqdm(range(args.source_epochs), desc="Meta-training")
    
    for epoch in progress_bar:
        start_time = time.time()
        
        meta_loss = model.meta_train_revise(
            *source_dataset.get_maml_task_batch(config['task']['task_num'])
        )
        
        training_losses.append(meta_loss)
        model.scheduler.step()
        
        progress_bar.set_postfix({
            'Meta Loss': f'{meta_loss:.6f}',
            'Best Loss': f'{best_meta_loss:.6f}',
            'LR': f'{model.meta_optim.param_groups[0]["lr"]:.2e}',
            'Epoch': f'{epoch+1}/{args.source_epochs}'
        })
        
        if epoch % args.log_interval == 0:
            log_metrics({
                'meta_loss': meta_loss,
                'learning_rate': model.meta_optim.param_groups[0]['lr']
            }, epoch, 'meta_train', log_file)
        
        if epoch % args.save_interval == 0 and epoch > 0:
            checkpoint_path = os.path.join(exp_dir, 'checkpoints', f'meta_checkpoint_epoch_{epoch}.pth')
            save_maml_checkpoint(model, epoch, meta_loss, checkpoint_path)
        
        if meta_loss < best_meta_loss:
            best_meta_loss = meta_loss
            best_model_path = os.path.join(exp_dir, 'models', 'best_meta_model.pth')
            save_maml_checkpoint(model, epoch, meta_loss, best_model_path)
        
        # Early stopping with minimum epoch constraint
        if early_stopping(meta_loss, model.model, epoch):
            print(f"\nEarly stopping at epoch {epoch} (after minimum {args.source_epochs} epochs)")
            break
    
    print(f"\nMeta-training completed. Best loss: {best_meta_loss:.6f}")
    print(f"Training completed at epoch {epoch+1}/{args.source_epochs}")
    return training_losses, best_meta_loss


def finetune_phase(model: WaveTrafficMAML, target_dataloader, test_dataloader, 
                  args, exp_dir: str):
    print("\n" + "="*60)
    print("STARTING FINE-TUNING PHASE")
    print("="*60)
    
    model.finetuning(target_dataloader, test_dataloader, args.target_epochs)


def evaluation_phase(model: WaveTrafficMAML, test_dataloader, exp_dir: str):
    print("\n" + "="*60)
    print("STARTING FINAL EVALUATION")
    print("="*60)
    
    eval_results = model.evaluate(test_dataloader)
    
    result_print(eval_results['metrics'], 'Final Test Results')
    
    save_all_plots(
        exp_dir,
        eval_results.get('model_outputs', {}),
        {'Final': eval_results['metrics']},
        model.loss_history.get('total_loss', []),
        predictions=eval_results['predictions'],
        targets=eval_results['targets']
    )
    
    return eval_results


def save_final_results(exp_dir: str, args, config: Dict, training_losses: List[float],
                      best_meta_loss: float, eval_results: Dict):
    
    # Ensure the results directory exists
    os.makedirs(exp_dir, exist_ok=True)
    
    final_results = {
        'experiment_config': {
            'args': vars(args),
            'config': config,
            'experiment_dir': exp_dir,
            'save_directory': args.save_dir
        },
        'training_results': {
            'best_meta_loss': best_meta_loss,
            'training_losses': training_losses,
            'total_epochs': len(training_losses),
            'convergence_epoch': training_losses.index(min(training_losses)) + 1 if training_losses else 0
        },
        'evaluation_results': {
            'final_metrics': eval_results['metrics'],
            'avg_mae': float(np.mean(eval_results['metrics']['MAE'][:3])),
            'avg_rmse': float(np.mean(eval_results['metrics']['RMSE'][:3])),
            'avg_mape': float(np.mean(eval_results['metrics']['MAPE'][:3]) * 100),
            'best_mae': float(np.min(eval_results['metrics']['MAE'])),
            'best_rmse': float(np.min(eval_results['metrics']['RMSE'])),
            'best_mape': float(np.min(eval_results['metrics']['MAPE']) * 100)
        },
        'model_info': {
            'wave_components': args.wave_components,
            'frequency_bands': args.frequency_bands,
            'kernel_sizes': [3, 5, 7],
            'target_dataset': args.test_dataset,
            'hidden_dim': config['model'].get('hidden_dim', 128),
            'prediction_horizon': config['task'].get('pred_num', 6)
        }
    }
    
    # Save comprehensive results
    results_file = os.path.join(exp_dir, 'final_results.yaml')
    save_config(final_results, results_file)
    
    # Save raw data
    predictions_file = os.path.join(exp_dir, 'predictions.npy')
    targets_file = os.path.join(exp_dir, 'targets.npy')
    losses_file = os.path.join(exp_dir, 'training_losses.npy')
    metrics_file = os.path.join(exp_dir, 'metrics.npy')
    
    np.save(predictions_file, eval_results['predictions'])
    np.save(targets_file, eval_results['targets'])
    np.save(losses_file, np.array(training_losses))
    np.save(metrics_file, eval_results['metrics'])
    
    # Create a summary report
    summary_file = os.path.join(exp_dir, 'experiment_summary.txt')
    with open(summary_file, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("WAVE TRAFFIC FLOW PREDICTION - EXPERIMENT SUMMARY\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Dataset: {args.test_dataset}\n")
        f.write(f"Wave Components: {args.wave_components}\n")
        f.write(f"Frequency Bands: {args.frequency_bands}\n")
        f.write(f"Kernel Sizes: [3, 5, 7]\n")
        f.write(f"Training Epochs: {len(training_losses)}\n")
        f.write(f"Target Days: {args.target_days}\n\n")
        
        f.write("PERFORMANCE METRICS:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Average MAE:  {final_results['evaluation_results']['avg_mae']:.4f}\n")
        f.write(f"Average RMSE: {final_results['evaluation_results']['avg_rmse']:.4f}\n")
        f.write(f"Average MAPE: {final_results['evaluation_results']['avg_mape']:.2f}%\n\n")
        
        f.write(f"Best MAE:     {final_results['evaluation_results']['best_mae']:.4f}\n")
        f.write(f"Best RMSE:    {final_results['evaluation_results']['best_rmse']:.4f}\n")
        f.write(f"Best MAPE:    {final_results['evaluation_results']['best_mape']:.2f}%\n\n")
        
        f.write("TRAINING DETAILS:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Best Meta Loss: {best_meta_loss:.6f}\n")
        f.write(f"Convergence Epoch: {final_results['training_results']['convergence_epoch']}\n")
        f.write(f"Final Loss: {training_losses[-1]:.6f}\n\n")
        
        f.write("FILES SAVED:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Results: {results_file}\n")
        f.write(f"Predictions: {predictions_file}\n")
        f.write(f"Targets: {targets_file}\n")
        f.write(f"Training Losses: {losses_file}\n")
        f.write(f"Metrics: {metrics_file}\n")
        f.write(f"Plots Directory: {os.path.join(exp_dir, 'plots')}\n")
    
    print(f"\n" + "=" * 60)
    print("RESULTS SAVED TO:", exp_dir)
    print("=" * 60)
    print(f"Main Results: {results_file}")
    print(f"Summary: {summary_file}")
    print(f"Plots: {os.path.join(exp_dir, 'plots')}")
    print(f"Raw Data: predictions.npy, targets.npy, training_losses.npy")
    print("=" * 60)
    
    print("\nFinal Performance Summary:")
    print(f"Average MAE:  {final_results['evaluation_results']['avg_mae']:.4f}")
    print(f"Average RMSE: {final_results['evaluation_results']['avg_rmse']:.4f}")
    print(f"Average MAPE: {final_results['evaluation_results']['avg_mape']:.2f}%")
    print(f"Best MAE:     {final_results['evaluation_results']['best_mae']:.4f}")
    print(f"Best RMSE:    {final_results['evaluation_results']['best_rmse']:.4f}")
    print(f"Best MAPE:    {final_results['evaluation_results']['best_mape']:.2f}%")


def main():
    print("Wave Traffic Flow Prediction with MAML")
    print("="*60)
    
    args = parse_arguments()
    config, exp_dir, device = setup_experiment(args)
    source_dataset, target_dataloader, test_dataloader = create_datasets(config, args, device)
    model = create_model_and_optimizer(config, args, device)
    
    training_losses, best_meta_loss = meta_train_phase(
        model, source_dataset, config, args, exp_dir, device
    )
    finetune_phase(model, target_dataloader, test_dataloader, args, exp_dir)
    
    eval_results = evaluation_phase(model, test_dataloader, exp_dir)
    
    save_final_results(exp_dir, args, config, training_losses, best_meta_loss, eval_results)
    
    print("\n" + "="*60)
    print("EXPERIMENT COMPLETED SUCCESSFULLY")
    print("="*60)

if __name__ == '__main__':
    main()