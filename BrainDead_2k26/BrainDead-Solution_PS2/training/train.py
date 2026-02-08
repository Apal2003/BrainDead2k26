"""
Training Loops and Loss Functions
Complete training pipeline for Cognitive Radiology Report Generator
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import numpy as np
from pathlib import Path


class CombinedLoss(nn.Module):
    """
    Combined loss function for multi-task learning:
    1. Classification loss (BCE for multi-label)
    2. Generation loss (Cross-entropy for text)
    """
    
    def __init__(self, cls_weight=0.3, gen_weight=0.7):
        super().__init__()
        self.cls_weight = cls_weight
        self.gen_weight = gen_weight
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding
    
    def forward(self, cls_logits, gen_logits, cls_labels, gen_labels):
        """
        Args:
            cls_logits: Classification logits (B, num_classes)
            gen_logits: Generation logits (B, seq_len, vocab_size)
            cls_labels: Classification labels (B, num_classes)
            gen_labels: Generation labels (B, seq_len)
        
        Returns:
            total_loss: Weighted combination of losses
            loss_dict: Dictionary with individual losses
        """
        # Classification loss
        cls_loss = self.bce_loss(cls_logits, cls_labels)
        
        # Generation loss
        gen_logits_flat = gen_logits.view(-1, gen_logits.size(-1))
        gen_labels_flat = gen_labels.view(-1)
        gen_loss = self.ce_loss(gen_logits_flat, gen_labels_flat)
        
        # Combined loss
        total_loss = self.cls_weight * cls_loss + self.gen_weight * gen_loss
        
        loss_dict = {
            'total': total_loss.item(),
            'classification': cls_loss.item(),
            'generation': gen_loss.item()
        }
        
        return total_loss, loss_dict


class Trainer:
    """
    Complete training pipeline
    """
    
    def __init__(self, 
                 model,
                 train_loader,
                 val_loader,
                 optimizer,
                 scheduler,
                 device,
                 config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.config = config
        
        # Loss function
        self.criterion = CombinedLoss(
            cls_weight=config.get('cls_weight', 0.3),
            gen_weight=config.get('gen_weight', 0.7)
        )
        
        # Tracking
        self.best_val_loss = float('inf')
        self.history = {
            'train_loss': [],
            'train_cls_loss': [],
            'train_gen_loss': [],
            'val_loss': [],
            'val_cls_loss': [],
            'val_gen_loss': []
        }
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        total_cls_loss = 0
        total_gen_loss = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.config["num_epochs"]} [Train]')
        
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            images = batch['image'].to(self.device)
            labels = batch['labels'].to(self.device)
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            cls_logits, gen_logits = self.model(
                images=images,
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            # Compute loss
            loss, loss_dict = self.criterion(
                cls_logits=cls_logits,
                gen_logits=gen_logits,
                cls_labels=labels,
                gen_labels=input_ids  # Teacher forcing
            )
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Track losses
            total_loss += loss_dict['total']
            total_cls_loss += loss_dict['classification']
            total_gen_loss += loss_dict['generation']
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss_dict['total']:.4f}",
                'cls': f"{loss_dict['classification']:.4f}",
                'gen': f"{loss_dict['generation']:.4f}"
            })
        
        # Update scheduler
        self.scheduler.step()
        
        # Compute average losses
        num_batches = len(self.train_loader)
        avg_loss = total_loss / num_batches
        avg_cls_loss = total_cls_loss / num_batches
        avg_gen_loss = total_gen_loss / num_batches
        
        return avg_loss, avg_cls_loss, avg_gen_loss
    
    def validate(self, epoch):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        total_cls_loss = 0
        total_gen_loss = 0
        
        pbar = tqdm(self.val_loader, desc=f'Epoch {epoch+1}/{self.config["num_epochs"]} [Val]')
        
        with torch.no_grad():
            for batch in pbar:
                # Move to device
                images = batch['image'].to(self.device)
                labels = batch['labels'].to(self.device)
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                # Forward pass
                cls_logits, gen_logits = self.model(
                    images=images,
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                # Compute loss
                loss, loss_dict = self.criterion(
                    cls_logits=cls_logits,
                    gen_logits=gen_logits,
                    cls_labels=labels,
                    gen_labels=input_ids
                )
                
                # Track losses
                total_loss += loss_dict['total']
                total_cls_loss += loss_dict['classification']
                total_gen_loss += loss_dict['generation']
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f"{loss_dict['total']:.4f}",
                    'cls': f"{loss_dict['classification']:.4f}",
                    'gen': f"{loss_dict['generation']:.4f}"
                })
        
        # Compute average losses
        num_batches = len(self.val_loader)
        avg_loss = total_loss / num_batches
        avg_cls_loss = total_cls_loss / num_batches
        avg_gen_loss = total_gen_loss / num_batches
        
        return avg_loss, avg_cls_loss, avg_gen_loss
    
    def train(self):
        """Complete training loop"""
        print("Starting training...")
        print(f"Device: {self.device}")
        print(f"Train batches: {len(self.train_loader)}")
        print(f"Val batches: {len(self.val_loader)}")
        print("=" * 80)
        
        for epoch in range(self.config['num_epochs']):
            # Train
            train_loss, train_cls_loss, train_gen_loss = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_cls_loss, val_gen_loss = self.validate(epoch)
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['train_cls_loss'].append(train_cls_loss)
            self.history['train_gen_loss'].append(train_gen_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_cls_loss'].append(val_cls_loss)
            self.history['val_gen_loss'].append(val_gen_loss)
            
            # Print epoch summary
            print(f"\nEpoch {epoch+1}/{self.config['num_epochs']} Summary:")
            print(f"  Train Loss: {train_loss:.4f} (Cls: {train_cls_loss:.4f}, Gen: {train_gen_loss:.4f})")
            print(f"  Val Loss:   {val_loss:.4f} (Cls: {val_cls_loss:.4f}, Gen: {val_gen_loss:.4f})")
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint(epoch, is_best=True)
                print(f"  ✓ New best model saved! (Val Loss: {val_loss:.4f})")
            
            # Save regular checkpoint
            if (epoch + 1) % self.config.get('save_every', 5) == 0:
                self.save_checkpoint(epoch, is_best=False)
            
            print("=" * 80)
        
        print("Training complete!")
        return self.history
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint"""
        checkpoint_dir = Path(self.config['output_path']) / 'models'
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'history': self.history,
            'config': self.config
        }
        
        if is_best:
            path = checkpoint_dir / 'best_model.pt'
        else:
            path = checkpoint_dir / f'checkpoint_epoch_{epoch+1}.pt'
        
        torch.save(checkpoint, path)


def create_trainer(model, train_loader, val_loader, config):
    """
    Create trainer with optimizer and scheduler
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config.get('weight_decay', 0.01)
    )
    
    # Scheduler
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config['num_epochs'],
        eta_min=config.get('min_lr', 1e-6)
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        config=config
    )
    
    return trainer