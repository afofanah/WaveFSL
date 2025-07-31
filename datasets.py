import torch
from torch_geometric.data import Data, Dataset, DataLoader
import numpy as np
import random
import os
from typing import Dict, List, Tuple, Optional, Any


class BBDefinedError(Exception):
    def __init__(self, ErrorInfo):
        super().__init__(self)
        self.errorinfo = ErrorInfo
    
    def __str__(self):
        return self.errorinfo


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


class WaveTrafficDataset(Dataset):
    
    def __init__(self, 
                 data_args: Dict, 
                 task_args: Dict, 
                 stage: str = 'source', 
                 test_data: str = 'metr-la', 
                 add_target: bool = True, 
                 target_days: int = 3):
        super(WaveTrafficDataset, self).__init__()
        
        self.data_args = data_args
        self.task_args = task_args
        self.his_num = task_args['his_num']
        self.pred_num = task_args['pred_num']
        self.stage = stage
        self.add_target = add_target
        self.test_data = test_data
        self.target_days = target_days
        
        self.A_list = {}
        self.edge_index_list = {}
        self.edge_attr_list = {}
        self.node_feature_list = {}
        self.x_list = {}
        self.y_list = {}
        self.means_list = {}
        self.stds_list = {}
        
        self.load_data(stage, test_data)
        
        if self.add_target:
            self.data_list = np.append(self.data_list, self.test_data)

    def load_data(self, stage: str, test_data: str):
        data_keys = np.array(self.data_args['data_keys'])
        if stage == 'source':
            self.data_list = np.delete(data_keys, np.where(data_keys == test_data))
        elif stage in ['target', 'target_maml', 'test', 'dann']:
            self.data_list = np.array([test_data])
        else:
            raise BBDefinedError(f'Error: Unsupported Stage: {stage}')

        for dataset_name in self.data_list:
            self._load_dataset(dataset_name, stage)
        
        if stage == 'source' and self.add_target:
            self._load_dataset(test_data, 'target')

    def _load_dataset(self, dataset_name: str, stage: str):
        dataset_config = self.data_args[dataset_name]
        
        A = np.load(dataset_config['adjacency_matrix_path'])
        edge_index, edge_attr, node_feature = self.get_attr_func(
            dataset_config['adjacency_matrix_path']
        )
        
        self.A_list[dataset_name] = torch.from_numpy(get_normalized_adj(A))
        self.edge_index_list[dataset_name] = edge_index
        self.edge_attr_list[dataset_name] = edge_attr
        self.node_feature_list[dataset_name] = node_feature

        X = np.load(dataset_config['dataset_path'])
        X = X.transpose((1, 2, 0))
        X = X.astype(np.float32)
        
        means = np.mean(X, axis=(0, 2))
        X = X - means.reshape(1, -1, 1)
        stds = np.std(X, axis=(0, 2))
        stds = np.where(stds == 0, 1, stds)
        X = X / stds.reshape(1, -1, 1)

        if stage == 'source' or stage == 'dann':
            X_processed = X
        elif stage in ['target', 'target_maml']:
            X_processed = X[:, :, :288 * self.target_days]
        elif stage == 'test':
            X_processed = X[:, :, int(X.shape[2] * 0.8):]
        else:
            raise BBDefinedError(f'Error: Unsupported Stage: {stage}')
        
        x_inputs, y_outputs = generate_dataset(
            X_processed, self.his_num, self.pred_num, means, stds
        )
        
        self.x_list[dataset_name] = x_inputs.float()
        self.y_list[dataset_name] = y_outputs.float()
        self.means_list[dataset_name] = means
        self.stds_list[dataset_name] = stds

    def get_attr_func(self, matrix_path: str, 
                     edge_feature_matrix_path: Optional[str] = None,
                     node_feature_path: Optional[str] = None) -> Tuple:
        a, b = [], []
        edge_attr = []
        node_feature = None
        
        matrix = np.load(matrix_path)
        
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if matrix[i][j] > 0:
                    a.append(i)
                    b.append(j)
                    edge_attr.append(matrix[i][j])
        
        edge = [a, b]
        edge_index = torch.tensor(edge, dtype=torch.long)
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        
        return edge_index, edge_attr, node_feature

    def create_wave_data(self, x_data: torch.Tensor, y_data: torch.Tensor, 
                        select_dataset: str) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, nodes, time_steps, features = x_data.shape
        
        support_data = x_data.permute(0, 2, 1, 3).contiguous()
        support_data = support_data.view(batch_size, time_steps, nodes * features)
        
        if y_data.dim() == 2:
            y_data = y_data.unsqueeze(-1)
        
        query_data = y_data.permute(0, 2, 1).contiguous()
        
        return support_data, query_data

    def __getitem__(self, index: int) -> Tuple[Data, torch.Tensor]:
        if self.stage == 'source':
            select_dataset = random.choice(self.data_list)
            batch_size = self.task_args['batch_size']
            permutation = torch.randperm(self.x_list[select_dataset].shape[0])
            indices = permutation[:batch_size]
            x_data = self.x_list[select_dataset][indices]
            y_data = self.y_list[select_dataset][indices]
        
        elif self.stage == 'target_maml':
            select_dataset = self.data_list[0]
            batch_size = self.task_args['batch_size']
            permutation = torch.randperm(self.x_list[select_dataset].shape[0])
            indices = permutation[:batch_size]
            x_data = self.x_list[select_dataset][indices]
            y_data = self.y_list[select_dataset][indices]
        
        else:
            select_dataset = self.data_list[0]
            x_data = self.x_list[select_dataset][index:index + 1]
            y_data = self.y_list[select_dataset][index:index + 1]

        node_num = self.A_list[select_dataset].shape[0]
        data_i = Data(node_num=node_num, x=x_data, y=y_data)
        data_i.edge_index = self.edge_index_list[select_dataset]
        data_i.data_name = select_dataset
        
        A_wave = self.A_list[select_dataset]
        
        return data_i, A_wave

    def get_maml_task_batch(self, task_num: int) -> Tuple[List, List, List, List]:
        spt_task_data, qry_task_data = [], []
        spt_task_A_wave, qry_task_A_wave = [], []

        select_dataset = random.choice(self.data_list)
        batch_size = self.task_args['batch_size']

        for i in range(task_num * 2):
            permutation = torch.randperm(self.x_list[select_dataset].shape[0])
            indices = permutation[:batch_size]
            x_data = self.x_list[select_dataset][indices]
            y_data = self.y_list[select_dataset][indices]
            
            node_num = self.A_list[select_dataset].shape[0]
            data_i = Data(node_num=node_num, x=x_data, y=y_data)
            data_i.edge_index = self.edge_index_list[select_dataset]
            data_i.data_name = select_dataset
            
            A_wave = self.A_list[select_dataset].float()
            
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            if i % 2 == 0:
                spt_task_data.append(data_i.to(device))
                spt_task_A_wave.append(A_wave.to(device))
            else:
                qry_task_data.append(data_i.to(device))
                qry_task_A_wave.append(A_wave.to(device))

        return spt_task_data, spt_task_A_wave, qry_task_data, qry_task_A_wave

    def get_wave_task_batch(self, task_num: int) -> Tuple[List, List]:
        spt_task_data, _, qry_task_data, _ = self.get_maml_task_batch(task_num)
        
        support_batches = []
        query_batches = []
        
        for spt_data, qry_data in zip(spt_task_data, qry_task_data):
            spt_support, spt_query = self.create_wave_data(spt_data.x, spt_data.y, spt_data.data_name)
            qry_support, qry_query = self.create_wave_data(qry_data.x, qry_data.y, qry_data.data_name)
            
            support_batches.append((spt_support, spt_query))
            query_batches.append((qry_support, qry_query))
        
        return support_batches, query_batches

    def __len__(self) -> int:
        if self.stage == 'source':
            return 100000000
        else:
            data_length = self.x_list[self.data_list[0]].shape[0]
            return data_length

    def get_dataset_info(self, dataset_name: str) -> Dict:
        if dataset_name not in self.data_args:
            raise ValueError(f"Dataset {dataset_name} not found in data_args")
        
        config = self.data_args[dataset_name]
        return {
            'name': dataset_name,
            'node_num': config['node_num'],
            'time_step': config['time_step'],
            'speed_mean': config['speed_mean'],
            'speed_std': config['speed_std'],
            'adjacency_shape': self.A_list.get(dataset_name, torch.tensor([])).shape,
            'data_samples': self.x_list.get(dataset_name, torch.tensor([])).shape[0] if dataset_name in self.x_list else 0
        }

    def get_all_datasets_info(self) -> Dict:
        info = {}
        for dataset_name in self.data_list:
            info[dataset_name] = self.get_dataset_info(dataset_name)
        return info


class WaveDataManager:
    def __init__(self, data_args: Dict, task_args: Dict):
        self.data_args = data_args
        self.task_args = task_args
        self.datasets = {}
    
    def create_dataset(self, stage: str, test_data: str, **kwargs) -> WaveTrafficDataset:
        dataset = WaveTrafficDataset(
            data_args=self.data_args,
            task_args=self.task_args,
            stage=stage,
            test_data=test_data,
            **kwargs
        )
        self.datasets[f"{stage}_{test_data}"] = dataset
        return dataset
    
    def create_dataloaders(self, test_data: str, target_days: int = 3) -> Dict[str, Any]:
        dataloaders = {}
        
        source_dataset = self.create_dataset('source', test_data)
        dataloaders['source'] = DataLoader(
            source_dataset,
            batch_size=self.task_args['batch_size'],
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        target_dataset = self.create_dataset('target', test_data, target_days=target_days)
        dataloaders['target'] = DataLoader(
            target_dataset,
            batch_size=self.task_args['batch_size'],
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        test_dataset = self.create_dataset('test', test_data)
        dataloaders['test'] = DataLoader(
            test_dataset,
            batch_size=self.task_args['test_batch_size'],
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        return dataloaders
    
    def get_dataset_statistics(self) -> Dict:
        stats = {}
        for key, dataset in self.datasets.items():
            stats[key] = dataset.get_all_datasets_info()
        return stats


class DataAugmentation:
    def __init__(self, noise_std: float = 0.01, rotation_prob: float = 0.1):
        self.noise_std = noise_std
        self.rotation_prob = rotation_prob
    
    def add_gaussian_noise(self, x: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(x) * self.noise_std
        return x + noise
    
    def temporal_shift(self, x: torch.Tensor, max_shift: int = 2) -> torch.Tensor:
        batch_size, time_steps, features = x.shape
        shift = random.randint(-max_shift, max_shift)
        
        if shift > 0:
            shifted_x = torch.cat([x[:, shift:], x[:, :shift]], dim=1)
        elif shift < 0:
            shifted_x = torch.cat([x[:, -shift:], x[:, :-shift]], dim=1)
        else:
            shifted_x = x
        
        return shifted_x
    
    def feature_dropout(self, x: torch.Tensor, dropout_prob: float = 0.1) -> torch.Tensor:
        mask = torch.rand(x.shape[-1]) > dropout_prob
        return x * mask.float()
    
    def apply_augmentation(self, x: torch.Tensor) -> torch.Tensor:
        if random.random() < 0.5:
            x = self.add_gaussian_noise(x)
        
        if random.random() < 0.3:
            x = self.temporal_shift(x)
        
        if random.random() < 0.2:
            x = self.feature_dropout(x)
        
        return x


class AdaptiveDataLoader:
    def __init__(self, dataset: WaveTrafficDataset, batch_size: int, 
                 shuffle: bool = True, augmentation: bool = False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augmentation = DataAugmentation() if augmentation else None
        
    def __iter__(self):
        if self.shuffle:
            indices = torch.randperm(len(self.dataset))
        else:
            indices = torch.arange(len(self.dataset))
        
        for i in range(0, len(indices), self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            batch_data = []
            batch_A = []
            
            for idx in batch_indices:
                data, A = self.dataset[idx]
                
                if self.augmentation and self.dataset.stage == 'source':
                    data.x = self.augmentation.apply_augmentation(data.x)
                
                batch_data.append(data)
                batch_A.append(A)
            
            yield batch_data, batch_A
    
    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size


class BalancedSampler:
    def __init__(self, dataset: WaveTrafficDataset, samples_per_dataset: int = 100):
        self.dataset = dataset
        self.samples_per_dataset = samples_per_dataset
        
    def get_balanced_batch(self):
        balanced_data = []
        balanced_A = []
        
        for dataset_name in self.dataset.data_list:
            if dataset_name in self.dataset.x_list:
                n_samples = min(self.samples_per_dataset, 
                              self.dataset.x_list[dataset_name].shape[0])
                
                indices = torch.randperm(self.dataset.x_list[dataset_name].shape[0])[:n_samples]
                
                for idx in indices:
                    x_data = self.dataset.x_list[dataset_name][idx:idx+1]
                    y_data = self.dataset.y_list[dataset_name][idx:idx+1]
                    
                    node_num = self.dataset.A_list[dataset_name].shape[0]
                    data_i = Data(node_num=node_num, x=x_data, y=y_data)
                    data_i.edge_index = self.dataset.edge_index_list[dataset_name]
                    data_i.data_name = dataset_name
                    
                    balanced_data.append(data_i)
                    balanced_A.append(self.dataset.A_list[dataset_name])
        
        return balanced_data, balanced_A


class DatasetValidator:
    @staticmethod
    def validate_dataset(dataset: WaveTrafficDataset) -> Dict[str, bool]:
        validation_results = {}
        
        for dataset_name in dataset.data_list:
            results = {
                'has_adjacency_matrix': dataset_name in dataset.A_list,
                'has_edge_index': dataset_name in dataset.edge_index_list,
                'has_input_data': dataset_name in dataset.x_list,
                'has_target_data': dataset_name in dataset.y_list,
                'data_shape_consistency': False,
                'no_nan_values': False,
                'positive_adjacency': False
            }
            
            if results['has_input_data'] and results['has_target_data']:
                x_shape = dataset.x_list[dataset_name].shape
                y_shape = dataset.y_list[dataset_name].shape
                results['data_shape_consistency'] = (x_shape[0] == y_shape[0])
            
            if results['has_input_data']:
                x_data = dataset.x_list[dataset_name]
                results['no_nan_values'] = not torch.isnan(x_data).any()
            
            if results['has_adjacency_matrix']:
                A = dataset.A_list[dataset_name]
                results['positive_adjacency'] = (A >= 0).all()
            
            validation_results[dataset_name] = results
        
        return validation_results
    
    @staticmethod
    def print_validation_report(validation_results: Dict[str, Dict[str, bool]]):
        for dataset_name, results in validation_results.items():
            print(f"\n=== {dataset_name} Validation ===")
            for check, passed in results.items():
                status = "✓" if passed else "✗"
                print(f"{status} {check.replace('_', ' ').title()}")


def Traffic_data(*args, **kwargs):
    return WaveTrafficDataset(*args, **kwargs)