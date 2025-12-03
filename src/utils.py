"""
Utility functions for the Hybrid VAE recommendation system.

This module contains helper functions for data loading, model management,
and common operations used throughout the system.
"""

import torch
import numpy as np
import pandas as pd
import pickle
import json
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from scipy.sparse import csr_matrix, save_npz, load_npz
import shutil
import matplotlib.pyplot as plt
import seaborn as sns

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_logging(log_level: str = "INFO", 
                 log_file: Optional[str] = None) -> None:
    """
    Setup logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional log file path
    """
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=handlers,
        force=True
    )


def save_json(data: Any, filepath: str, indent: int = 2) -> None:
    """
    Save data to JSON file.
    
    Args:
        data: Data to save
        filepath: Output file path
        indent: JSON indentation
    """
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=indent, default=str)


def load_json(filepath: str) -> Any:
    """
    Load data from JSON file.
    
    Args:
        filepath: Input file path
        
    Returns:
        Loaded data
    """
    with open(filepath, 'r') as f:
        return json.load(f)


def save_pickle(data: Any, filepath: str) -> None:
    """
    Save data to pickle file.
    
    Args:
        data: Data to save
        filepath: Output file path
    """
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)


def load_pickle(filepath: str) -> Any:
    """
    Load data from pickle file.
    
    Args:
        filepath: Input file path
        
    Returns:
        Loaded data
    """
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def save_config(config: Dict, filepath: str) -> None:
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        filepath: Output file path
    """
    with open(filepath, 'w') as f:
        yaml.dump(config, f, indent=2)


def load_config(filepath: str) -> Dict:
    """
    Load configuration from YAML file.
    
    Args:
        filepath: Input file path
        
    Returns:
        Configuration dictionary
    """
    with open(filepath, 'r') as f:
        return yaml.safe_load(f)


def ensure_dir(directory: Union[str, Path]) -> Path:
    """
    Ensure directory exists, create if it doesn't.
    
    Args:
        directory: Directory path
        
    Returns:
        Path object
    """
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_device(preferred_device: Optional[str] = None) -> torch.device:
    """
    Get the best available device for computation.
    
    Args:
        preferred_device: Preferred device ('cuda' or 'cpu')
        
    Returns:
        PyTorch device
    """
    if preferred_device:
        device = torch.device(preferred_device)
        if device.type == 'cuda' and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available, using CPU")
            device = torch.device('cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    logger.info(f"Using device: {device}")
    return device


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """
    Count model parameters.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with parameter counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'non_trainable': total_params - trainable_params
    }


def save_sparse_matrix(matrix: csr_matrix, filepath: str) -> None:
    """
    Save sparse matrix to file.
    
    Args:
        matrix: Sparse matrix to save
        filepath: Output file path
    """
    save_npz(filepath, matrix)


def load_sparse_matrix(filepath: str) -> csr_matrix:
    """
    Load sparse matrix from file.
    
    Args:
        filepath: Input file path
        
    Returns:
        Loaded sparse matrix
    """
    return load_npz(filepath)


def compute_dataset_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute comprehensive dataset statistics.
    
    Args:
        df: DataFrame with user-item interactions
        
    Returns:
        Dictionary with statistics
    """
    stats = {
        'total_interactions': len(df),
        'unique_users': df['user_id'].nunique(),
        'unique_items': df['asin'].nunique(),
        'sparsity': 1 - len(df) / (df['user_id'].nunique() * df['asin'].nunique()),
        'avg_interactions_per_user': len(df) / df['user_id'].nunique(),
        'avg_interactions_per_item': len(df) / df['asin'].nunique(),
    }
    
    # User interaction distribution
    user_counts = df['user_id'].value_counts()
    stats['user_interaction_stats'] = {
        'min': user_counts.min(),
        'max': user_counts.max(),
        'mean': user_counts.mean(),
        'median': user_counts.median(),
        'std': user_counts.std()
    }
    
    # Item interaction distribution
    item_counts = df['asin'].value_counts()
    stats['item_interaction_stats'] = {
        'min': item_counts.min(),
        'max': item_counts.max(),
        'mean': item_counts.mean(),
        'median': item_counts.median(),
        'std': item_counts.std()
    }
    
    return stats


def plot_training_history(history: Dict[str, List[float]], 
                         save_path: Optional[str] = None) -> None:
    """
    Plot training history curves.
    
    Args:
        history: Dictionary with training history
        save_path: Optional path to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Total loss
    axes[0, 0].plot(history['train_losses'], label='Train')
    if 'val_losses' in history:
        axes[0, 0].plot(history['val_losses'], label='Validation')
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Reconstruction loss
    if 'train_recon_losses' in history:
        axes[0, 1].plot(history['train_recon_losses'], label='Reconstruction Loss')
        axes[0, 1].set_title('Reconstruction Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
    
    # KL divergence loss
    if 'train_kl_losses' in history:
        axes[1, 0].plot(history['train_kl_losses'], label='KL Loss')
        axes[1, 0].set_title('KL Divergence Loss')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
    
    # Combined view
    axes[1, 1].plot(history['train_losses'], label='Total Loss')
    if 'train_recon_losses' in history:
        axes[1, 1].plot(history['train_recon_losses'], label='Recon Loss')
    if 'train_kl_losses' in history:
        axes[1, 1].plot(history['train_kl_losses'], label='KL Loss')
    axes[1, 1].set_title('All Losses')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved training plot to {save_path}")
    
    plt.show()


def plot_dataset_statistics(stats: Dict[str, Any], 
                          save_path: Optional[str] = None) -> None:
    """
    Plot dataset statistics.
    
    Args:
        stats: Dataset statistics dictionary
        save_path: Optional path to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Basic stats
    basic_stats = [
        ('Total Interactions', stats['total_interactions']),
        ('Unique Users', stats['unique_users']),
        ('Unique Items', stats['unique_items']),
        ('Sparsity', f"{stats['sparsity']:.4f}")
    ]
    
    for i, (label, value) in enumerate(basic_stats):
        axes[0, 0].text(0.1, 0.8 - i * 0.2, f"{label}: {value}", 
                       fontsize=12, transform=axes[0, 0].transAxes)
    axes[0, 0].set_title('Dataset Overview')
    axes[0, 0].axis('off')
    
    # User interaction distribution (top users)
    user_stats = stats['user_interaction_stats']
    user_metrics = ['min', 'max', 'mean', 'median', 'std']
    user_values = [user_stats[metric] for metric in user_metrics]
    
    axes[0, 1].bar(user_metrics, user_values)
    axes[0, 1].set_title('User Interaction Statistics')
    axes[0, 1].set_ylabel('Number of Interactions')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Item interaction distribution
    item_stats = stats['item_interaction_stats']
    item_values = [item_stats[metric] for metric in user_metrics]
    
    axes[1, 0].bar(user_metrics, item_values)
    axes[1, 0].set_title('Item Interaction Statistics')
    axes[1, 0].set_ylabel('Number of Interactions')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Interaction density
    axes[1, 1].bar(['Interactions', 'Possible'], 
                  [stats['total_interactions'], 
                   stats['unique_users'] * stats['unique_items']])
    axes[1, 1].set_title('Interaction Matrix Density')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_yscale('log')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved statistics plot to {save_path}")
    
    plt.show()


def backup_model(model_path: str, backup_dir: str, keep_n: int = 5) -> None:
    """
    Create backup of model and manage backup retention.
    
    Args:
        model_path: Path to model file
        backup_dir: Directory to store backups
        keep_n: Number of backups to keep
    """
    backup_path = ensure_dir(backup_dir)
    
    # Create timestamp for backup
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create backup filename
    model_name = Path(model_path).stem
    backup_filename = f"{model_name}_backup_{timestamp}.pth"
    backup_filepath = backup_path / backup_filename
    
    # Copy model
    shutil.copy2(model_path, backup_filepath)
    logger.info(f"Created backup: {backup_filepath}")
    
    # Cleanup old backups
    backups = list(backup_path.glob(f"{model_name}_backup_*.pth"))
    backups.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    
    if len(backups) > keep_n:
        for old_backup in backups[keep_n:]:
            old_backup.unlink()
            logger.info(f"Removed old backup: {old_backup}")


def validate_data_consistency(data_dir: str) -> Dict[str, bool]:
    """
    Validate consistency between different data files.
    
    Args:
        data_dir: Directory containing dataset files
        
    Returns:
        Dictionary with validation results
    """
    data_path = Path(data_dir)
    results = {}
    
    try:
        # Load mappings
        mappings = load_pickle(data_path / 'mappings.pkl')
        user_to_idx = mappings['user_to_idx']
        item_to_idx = mappings['item_to_idx']
        
        # Load interaction matrix
        interaction_matrix = load_pickle(data_path / 'interaction_matrix.pkl')
        
        # Check dimensions
        results['matrix_dimensions'] = (
            interaction_matrix.shape[0] == len(user_to_idx) and
            interaction_matrix.shape[1] == len(item_to_idx)
        )
        
        # Load DataFrames
        train_df = pd.read_csv(data_path / 'train.csv')
        val_df = pd.read_csv(data_path / 'val.csv')
        test_df = pd.read_csv(data_path / 'test.csv')
        
        # Check if all users/items in DataFrames exist in mappings
        all_users = set(train_df['user_id']).union(val_df['user_id']).union(test_df['user_id'])
        all_items = set(train_df['asin']).union(val_df['asin']).union(test_df['asin'])
        
        results['users_in_mappings'] = all_users.issubset(set(user_to_idx.keys()))
        results['items_in_mappings'] = all_items.issubset(set(item_to_idx.keys()))
        
        # Check for overlaps between splits
        train_pairs = set(zip(train_df['user_id'], train_df['asin']))
        val_pairs = set(zip(val_df['user_id'], val_df['asin']))
        test_pairs = set(zip(test_df['user_id'], test_df['asin']))
        
        results['no_train_val_overlap'] = len(train_pairs.intersection(val_pairs)) == 0
        results['no_train_test_overlap'] = len(train_pairs.intersection(test_pairs)) == 0
        results['no_val_test_overlap'] = len(val_pairs.intersection(test_pairs)) == 0
        
    except Exception as e:
        logger.error(f"Error validating data consistency: {e}")
        results['validation_error'] = str(e)
    
    return results


def memory_usage_info() -> Dict[str, str]:
    """
    Get memory usage information.
    
    Returns:
        Dictionary with memory info
    """
    import psutil
    import gc
    
    # System memory
    memory = psutil.virtual_memory()
    
    # GPU memory if available
    gpu_info = {}
    if torch.cuda.is_available():
        gpu_info = {
            'gpu_allocated': f"{torch.cuda.memory_allocated() / 1e9:.2f} GB",
            'gpu_cached': f"{torch.cuda.memory_reserved() / 1e9:.2f} GB",
            'gpu_max_allocated': f"{torch.cuda.max_memory_allocated() / 1e9:.2f} GB"
        }
    
    return {
        'system_total': f"{memory.total / 1e9:.2f} GB",
        'system_available': f"{memory.available / 1e9:.2f} GB",
        'system_used': f"{memory.used / 1e9:.2f} GB",
        'system_percent': f"{memory.percent:.1f}%",
        **gpu_info
    }


class ExperimentTracker:
    """
    Simple experiment tracking utility.
    """
    
    def __init__(self, experiment_name: str, output_dir: str):
        self.experiment_name = experiment_name
        self.output_dir = ensure_dir(output_dir)
        self.experiment_dir = self.output_dir / experiment_name
        ensure_dir(self.experiment_dir)
        
        self.metrics = {}
        self.config = {}
        
    def log_config(self, config: Dict) -> None:
        """Log experiment configuration."""
        self.config.update(config)
        save_config(self.config, self.experiment_dir / 'config.yaml')
    
    def log_metric(self, name: str, value: float, step: Optional[int] = None) -> None:
        """Log a metric value."""
        if name not in self.metrics:
            self.metrics[name] = []
        
        entry = {'value': value}
        if step is not None:
            entry['step'] = step
            
        self.metrics[name].append(entry)
        
    def log_metrics(self, metrics_dict: Dict[str, float], step: Optional[int] = None) -> None:
        """Log multiple metrics."""
        for name, value in metrics_dict.items():
            self.log_metric(name, value, step)
    
    def save_metrics(self) -> None:
        """Save all metrics to file."""
        save_json(self.metrics, self.experiment_dir / 'metrics.json')
    
    def get_best_metric(self, metric_name: str, maximize: bool = False) -> Tuple[float, int]:
        """Get best value for a metric."""
        if metric_name not in self.metrics:
            raise ValueError(f"Metric '{metric_name}' not found")
        
        values = [entry['value'] for entry in self.metrics[metric_name]]
        
        if maximize:
            best_idx = np.argmax(values)
        else:
            best_idx = np.argmin(values)
        
        best_value = values[best_idx]
        best_step = self.metrics[metric_name][best_idx].get('step', best_idx)
        
        return best_value, best_step


# Export commonly used functions
__all__ = [
    'setup_logging', 'save_json', 'load_json', 'save_pickle', 'load_pickle',
    'save_config', 'load_config', 'ensure_dir', 'get_device', 'count_parameters',
    'save_sparse_matrix', 'load_sparse_matrix', 'compute_dataset_statistics',
    'plot_training_history', 'plot_dataset_statistics', 'backup_model',
    'validate_data_consistency', 'memory_usage_info', 'ExperimentTracker'
]