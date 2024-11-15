import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from pathlib import Path

class Trainer:
    def __init__(self, model, train_loader, val_loader, 
                 criterion, optimizer, device, save_dir: str = 'models'):
        """
        Initialize the trainer
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch
        """
        self.model.train()
        total_loss = 0
        predictions = []
        labels = []
        
        for batch_features, batch_labels in self.train_loader:
            # Move to device
            batch_features = batch_features.to(self.device)
            batch_labels = batch_labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(batch_features)
            loss = self.criterion(outputs, batch_labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Save metrics
            total_loss += loss.item()
            predictions.extend(outputs.argmax(dim=1).cpu().numpy())
            labels.extend(batch_labels.cpu().numpy())
        
        # Calculate metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='binary'
        )
        
        return {
            'loss': total_loss / len(self.train_loader),
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def validate(self) -> Dict[str, float]:
        """
        Validate the model
        """
        self.model.eval()
        total_loss = 0
        predictions = []
        labels = []
        
        with torch.no_grad():
            for batch_features, batch_labels in self.val_loader:
                batch_features = batch_features.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                outputs = self.model(batch_features)
                loss = self.criterion(outputs, batch_labels)
                
                total_loss += loss.item()
                predictions.extend(outputs.argmax(dim=1).cpu().numpy())
                labels.extend(batch_labels.cpu().numpy())
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='binary'
        )
        
        return {
            'val_loss': total_loss / len(self.val_loader),
            'val_precision': precision,
            'val_recall': recall,
            'val_f1': f1
        }
    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float], is_best: bool = False):
        """
        Save a model checkpoint
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics
        }
        
        # Save regular checkpoint
        torch.save(checkpoint, self.save_dir / f'checkpoint_epoch_{epoch}.pth')
        
        # Save best model if specified
        if is_best:
            torch.save(checkpoint, self.save_dir / 'best_model.pth')