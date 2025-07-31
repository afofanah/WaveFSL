import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.data import DataLoader
import numpy as np
import time
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict
import copy

from models.model_V2 import WaveFSL, WaveAwareLoss
from utils import metric_func, result_print, EarlyStopping, save_all_plots


class WaveTrafficMAML(nn.Module):
    def __init__(self, 
                 model: WaveFSL,
                 model_args: Dict,
                 task_args: Dict,
                 learning_strategy: str = 'FewShot'):
        super(WaveTrafficMAML, self).__init__()
        
        self.model = model
        self.model_args = model_args
        self.task_args = task_args
        self.learning_strategy = learning_strategy
        
        self.update_lr = model_args.get('update_lr', 0.01)
        self.meta_lr = model_args.get('meta_lr', 0.001)
        self.loss_lambda = model_args.get('loss_lambda', 1.5)
        self.update_step = model_args.get('update_step', 5)
        self.update_step_test = model_args.get('update_step_test', 10)
        
        self.loss_fn = WaveAwareLoss(
            prediction_weight=model_args.get('prediction_weight', 1.0),
            harmony_weight=model_args.get('harmony_weight', 0.1),
            interference_weight=model_args.get('interference_weight', 0.05),
            smoothness_weight=model_args.get('smoothness_weight', 0.02),
            consistency_weight=model_args.get('consistency_weight', 0.03)
        )
        
        self.meta_optim = optim.AdamW(self.model.parameters(), lr=self.meta_lr, 
                                     weight_decay=1e-4, betas=(0.9, 0.999))
    
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.meta_optim, T_0=50, T_mult=2, eta_min=1e-6
        )
        
        self.loss_history = defaultdict(list)
        
    def forward(self, data, A_wave):
        support_data, query_data = self._prepare_wave_data(data)
        outputs = self.model(support_data, query_data)
        
        predictions = outputs['predictions']
        meta_graph = A_wave
        
        return predictions, meta_graph
    
    def _prepare_wave_data(self, data):
        batch_size, nodes, time_steps, features = data.x.shape
        
        support_data = data.x.permute(0, 2, 1, 3).contiguous()
        support_data = support_data.view(batch_size, time_steps, nodes * features)
        
        query_data = support_data
        
        return support_data, query_data
    
    def fast_adapt(self, support_data, support_targets, adaptation_steps: int = None):
        if adaptation_steps is None:
            adaptation_steps = self.update_step
            
        fast_model = copy.deepcopy(self.model)
        fast_optimizer = optim.AdamW(fast_model.parameters(), lr=self.update_lr, 
                                    weight_decay=1e-5, betas=(0.9, 0.999))

        inner_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            fast_optimizer, T_max=adaptation_steps, eta_min=self.update_lr * 0.1
        )
        
        for step in range(adaptation_steps):
            fast_optimizer.zero_grad()
            
            outputs = fast_model(support_data, support_data)
            loss_dict = self.loss_fn(outputs, support_targets)
            loss = loss_dict['total_loss']
            
            loss.backward()
            
            # Gradient clipping for inner loop stability
            torch.nn.utils.clip_grad_norm_(fast_model.parameters(), max_norm=0.5)
            
            fast_optimizer.step()
            inner_scheduler.step()
        
        return fast_model
    
    def meta_train_revise(self, spt_task_data, spt_task_A, qry_task_data, qry_task_A):
        self.model.train()
        self.meta_optim.zero_grad()
        
        meta_loss = 0.0
        loss_components = defaultdict(float)
        
        # Gradient accumulation for more stable training
        accumulation_steps = max(1, len(spt_task_data) // 2)
        
        for i, (spt_data, qry_data) in enumerate(zip(spt_task_data, qry_task_data)):
            spt_support, _ = self._prepare_wave_data(spt_data)
            qry_support, _ = self._prepare_wave_data(qry_data)
            
            # Enhanced fast adaptation with more steps for better convergence
            adapted_model = self.fast_adapt(spt_support, spt_data.y, self.update_step + 2)
            
            outputs = adapted_model(qry_support, qry_support)
            loss_dict = self.loss_fn(outputs, qry_data.y)
            
            task_loss = loss_dict['total_loss'] / len(spt_task_data)
            meta_loss += task_loss
            
            # Backward pass for gradient accumulation
            task_loss.backward()
            
            for key, value in loss_dict.items():
                loss_components[key] += value.item()
            
            # Gradient clipping per task to prevent explosion
            if (i + 1) % accumulation_steps == 0 or i == len(spt_task_data) - 1:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.meta_optim.step()
                self.meta_optim.zero_grad()
        
        for key in loss_components:
            loss_components[key] /= len(spt_task_data)
            self.loss_history[key].append(loss_components[key])
        
        return meta_loss.item()
    
    def finetuning(self, target_dataloader, test_dataloader, epochs: int):
        self.model.train()
        
        finetune_optimizer = optim.Adam(self.model.parameters(), lr=self.update_lr, weight_decay=1e-4)
        finetune_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            finetune_optimizer, mode='min', factor=0.7, patience=15, verbose=True
        )
        
        # Set minimum epochs for fine-tuning (e.g., 250 epochs)
        early_stopping = EarlyStopping(patience=50, min_delta=0.00001, min_epochs=epochs)
        best_loss = float('inf')
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            for step, (data, A_wave) in enumerate(target_dataloader):
                if hasattr(data, '__iter__'):
                    data = data[0]
                    A_wave = A_wave[0]
                
                device = next(self.model.parameters()).device
                data = data.to(device)
                A_wave = A_wave.to(device)
                
                finetune_optimizer.zero_grad()
                
                support_data, query_data = self._prepare_wave_data(data)
                outputs = self.model(support_data, query_data)
                
                loss_dict = self.loss_fn(outputs, data.y)
                loss = loss_dict['total_loss']
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                finetune_optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            avg_loss = epoch_loss / max(num_batches, 1)
            finetune_scheduler.step(avg_loss)
            
            if avg_loss < best_loss:
                best_loss = avg_loss
            
            if epoch % 25 == 0:
                print(f"Finetune Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}, LR: {finetune_optimizer.param_groups[0]['lr']:.2e}")
                
                eval_results = self.evaluate(test_dataloader)
                result_print(eval_results['metrics'], f'Finetune Epoch {epoch+1}')
            
            # Early stopping with minimum epoch constraint
            if early_stopping(avg_loss, self.model, epoch):
                print(f"Early stopping at epoch {epoch+1} (after minimum {epochs} epochs)")
                break
    
    def evaluate(self, dataloader) -> Dict:
        self.model.eval()
        all_outputs = []
        all_targets = []
        
        with torch.no_grad():
            for step, (data, A_wave) in enumerate(dataloader):
                if hasattr(data, '__iter__'):
                    data = data[0]
                    A_wave = A_wave[0]
                
                device = next(self.model.parameters()).device
                data = data.to(device)
                A_wave = A_wave.to(device)
                
                support_data, query_data = self._prepare_wave_data(data)
                outputs = self.model(support_data, query_data)
                predictions = outputs['predictions']
                
                if step == 0:
                    all_outputs = predictions
                    all_targets = data.y
                else:
                    all_outputs = torch.cat((all_outputs, predictions))
                    all_targets = torch.cat((all_targets, data.y))
        
        outputs_np = all_outputs.permute(0, 2, 1).detach().cpu().numpy()
        targets_np = all_targets.permute(0, 2, 1).detach().cpu().numpy() if all_targets.dim() == 3 else all_targets.unsqueeze(-1).detach().cpu().numpy()
        
        pred_num = outputs_np.shape[1]
        result = metric_func(pred=outputs_np, y=targets_np, times=pred_num)
        
        return {
            'predictions': outputs_np,
            'targets': targets_np,
            'metrics': result,
            'model_outputs': {
                'predictions': all_outputs,
                'interference_pattern': outputs.get('interference_pattern'),
                'multiscale_features': outputs.get('multiscale_features')
            }
        }


class WaveTrafficTrainer:
    def __init__(self, 
                 model: WaveFSL,
                 config: Dict,
                 device: torch.device):
        self.model = model.to(device)
        self.config = config
        self.device = device
        
        self.loss_fn = WaveAwareLoss(
            prediction_weight=config['model'].get('prediction_weight', 1.0),
            harmony_weight=config['model'].get('harmony_weight', 0.1),
            interference_weight=config['model'].get('interference_weight', 0.05),
            smoothness_weight=config['model'].get('smoothness_weight', 0.02),
            consistency_weight=config['model'].get('consistency_weight', 0.03)
        )
        
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config['model'].get('meta_lr', 0.001),
            weight_decay=config.get('training', {}).get('weight_decay', 1e-4)
        )
        
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=50, T_mult=2, eta_min=1e-6
        )
        
        self.loss_history = defaultdict(list)
        self.metrics_history = defaultdict(list)
        
    def train_epoch(self, dataloader) -> Dict[str, float]:
        self.model.train()
        epoch_losses = defaultdict(float)
        num_batches = 0
        
        for batch_idx, (data, A_wave) in enumerate(dataloader):
            if hasattr(data, '__iter__'):
                data = data[0]
                A_wave = A_wave[0]
            
            data = data.to(self.device)
            A_wave = A_wave.to(self.device)
            
            self.optimizer.zero_grad()
            
            support_data, query_data = self._prepare_wave_data(data)
            outputs = self.model(support_data, query_data)
            
            loss_dict = self.loss_fn(outputs, data.y)
            total_loss = loss_dict['total_loss']
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            for key, value in loss_dict.items():
                epoch_losses[key] += value.item()
            
            num_batches += 1
        
        for key in epoch_losses:
            epoch_losses[key] /= max(num_batches, 1)
            self.loss_history[key].append(epoch_losses[key])
        
        return dict(epoch_losses)
    
    def validate_epoch(self, dataloader) -> Dict:
        self.model.eval()
        all_outputs = []
        all_targets = []
        val_losses = defaultdict(float)
        num_batches = 0
        
        with torch.no_grad():
            for batch_idx, (data, A_wave) in enumerate(dataloader):
                if hasattr(data, '__iter__'):
                    data = data[0]
                    A_wave = A_wave[0]
                
                data = data.to(self.device)
                A_wave = A_wave.to(self.device)
                
                support_data, query_data = self._prepare_wave_data(data)
                outputs = self.model(support_data, query_data)
                
                loss_dict = self.loss_fn(outputs, data.y)
                
                for key, value in loss_dict.items():
                    val_losses[key] += value.item()
                
                predictions = outputs['predictions']
                
                if batch_idx == 0:
                    all_outputs = predictions
                    all_targets = data.y
                else:
                    all_outputs = torch.cat((all_outputs, predictions))
                    all_targets = torch.cat((all_targets, data.y))
                
                num_batches += 1
        
        for key in val_losses:
            val_losses[key] /= max(num_batches, 1)
        
        outputs_np = all_outputs.permute(0, 2, 1).detach().cpu().numpy()
        targets_np = all_targets.permute(0, 2, 1).detach().cpu().numpy() if all_targets.dim() == 3 else all_targets.unsqueeze(-1).detach().cpu().numpy()
        
        pred_num = outputs_np.shape[1]
        metrics = metric_func(pred=outputs_np, y=targets_np, times=pred_num)
        
        return {
            'losses': dict(val_losses),
            'metrics': metrics,
            'predictions': outputs_np,
            'targets': targets_np
        }
    
    def _prepare_wave_data(self, data):
        batch_size, nodes, time_steps, features = data.x.shape
        
        support_data = data.x.permute(0, 2, 1, 3).contiguous()
        support_data = support_data.view(batch_size, time_steps, nodes * features)
        
        query_data = support_data
        
        return support_data, query_data
    
    def train(self, train_dataloader, val_dataloader, epochs: int, 
              save_dir: str = None) -> Dict:
        
        early_stopping = EarlyStopping(
            patience=self.config.get('training', {}).get('patience', 20),
            min_delta=0.001
        )
        
        best_val_loss = float('inf')
        training_start_time = time.time()
        
        for epoch in range(epochs):
            epoch_start_time = time.time()
            
            train_losses = self.train_epoch(train_dataloader)
            val_results = self.validate_epoch(val_dataloader)
            
            val_loss = val_results['losses']['total_loss']
            
            self.scheduler.step()
            
            epoch_time = time.time() - epoch_start_time
            
            if epoch % 20 == 0:
                print(f"Epoch {epoch:3d} | Train Loss: {train_losses['total_loss']:.6f} | "
                      f"Val Loss: {val_loss:.6f} | Time: {epoch_time:.2f}s")
                
                avg_mae = np.mean(val_results['metrics']['MAE'][:3])
                avg_rmse = np.mean(val_results['metrics']['RMSE'][:3])
                print(f"         | Val MAE: {avg_mae:.4f} | Val RMSE: {avg_rmse:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                if save_dir:
                    torch.save(self.model.state_dict(), f"{save_dir}/best_model.pth")
            
            if early_stopping(val_loss, self.model):
                print(f"Early stopping at epoch {epoch}")
                break
        
        total_training_time = time.time() - training_start_time
        
        final_results = {
            'training_time': total_training_time,
            'best_val_loss': best_val_loss,
            'loss_history': dict(self.loss_history),
            'final_val_results': val_results
        }
        
        if save_dir:
            save_all_plots(
                save_dir, 
                {'interference_pattern': None, 'multiscale_features': None},
                val_results['metrics'],
                self.loss_history['total_loss'],
                predictions=val_results['predictions'],
                targets=val_results['targets']
            )
        
        return final_results


def create_maml_model(config: Dict, args) -> WaveTrafficMAML:
    from models.model import create_adaptive_model_for_args
    
    base_model = create_adaptive_model_for_args(config, args)
    
    maml_model = WaveTrafficMAML(
        model=base_model,
        model_args=config['model'],
        task_args=config['task'],
        learning_strategy=getattr(args, 'learning_strategy', 'FewShot')
    )
    
    return maml_model


def save_maml_checkpoint(model: WaveTrafficMAML, epoch: int, loss: float, filepath: str):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.model.state_dict(),
        'meta_optim_state_dict': model.meta_optim.state_dict(),
        'loss': loss,
        'loss_history': dict(model.loss_history),
        'model_args': model.model_args,
        'task_args': model.task_args,
        'timestamp': time.time()
    }
    torch.save(checkpoint, filepath)


def load_maml_checkpoint(model: WaveTrafficMAML, filepath: str) -> Tuple[int, float]:
    checkpoint = torch.load(filepath)
    model.model.load_state_dict(checkpoint['model_state_dict'])
    model.meta_optim.load_state_dict(checkpoint['meta_optim_state_dict'])
    model.loss_history = defaultdict(list, checkpoint.get('loss_history', {}))
    
    return checkpoint['epoch'], checkpoint['loss']


class MetricTracker:
    def __init__(self):
        self.metrics = defaultdict(list)
        
    def update(self, new_metrics: Dict[str, float]):
        for key, value in new_metrics.items():
            self.metrics[key].append(value)
    
    def get_average(self, key: str, last_n: int = None) -> float:
        values = self.metrics[key]
        if last_n:
            values = values[-last_n:]
        return np.mean(values) if values else 0.0
    
    def get_best(self, key: str, mode: str = 'min') -> float:
        values = self.metrics[key]
        if not values:
            return float('inf') if mode == 'min' else float('-inf')
        return np.min(values) if mode == 'min' else np.max(values)
    
    def reset(self):
        self.metrics = defaultdict(list)


class LearningRateScheduler:
    def __init__(self, optimizer, schedule_type: str = 'cosine', **kwargs):
        self.optimizer = optimizer
        self.schedule_type = schedule_type
        
        if schedule_type == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=kwargs.get('T_max', 200), eta_min=kwargs.get('eta_min', 1e-6)
            )
        elif schedule_type == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                optimizer, step_size=kwargs.get('step_size', 50), gamma=kwargs.get('gamma', 0.5)
            )
        elif schedule_type == 'plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=kwargs.get('factor', 0.5), 
                patience=kwargs.get('patience', 10)
            )
        else:
            self.scheduler = None
    
    def step(self, metric=None):
        if self.scheduler is not None:
            if self.schedule_type == 'plateau':
                self.scheduler.step(metric)
            else:
                self.scheduler.step()
    
    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']


class GradientClipping:
    def __init__(self, max_norm: float = 1.0, norm_type: float = 2.0):
        self.max_norm = max_norm
        self.norm_type = norm_type
        
    def clip_gradients(self, model: nn.Module) -> float:
        return torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=self.max_norm, norm_type=self.norm_type
        ).item()


class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.model = model
        self.decay = decay
        self.ema_model = copy.deepcopy(model)
        self.ema_model.eval()
        
        for param in self.ema_model.parameters():
            param.requires_grad_(False)
    
    def update(self):
        with torch.no_grad():
            for ema_param, model_param in zip(self.ema_model.parameters(), self.model.parameters()):
                ema_param.data.mul_(self.decay).add_(model_param.data, alpha=1 - self.decay)
    
    def get_ema_model(self):
        return self.ema_model