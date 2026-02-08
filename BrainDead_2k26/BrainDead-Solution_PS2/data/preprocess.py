"""
Data Download and Preprocessing Scripts
For IU X-Ray and MIMIC-CXR datasets
"""

import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
from pathlib import Path


# CheXpert disease labels (14 pathologies)
CHEXPERT_LABELS = [
    'No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 
    'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation', 
    'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion', 
    'Pleural Other', 'Fracture', 'Support Devices'
]


def prepare_iu_dataset(df, image_dir):
    """
    Prepare IU X-Ray dataset with proper formatting
    
    Args:
        df: DataFrame with IU X-Ray data
        image_dir: Directory containing images
    
    Returns:
        processed_df: Processed DataFrame ready for training
    """
    print("Preparing IU X-Ray dataset...")
    
    # Combine findings and impression into full report
    df['report'] = df.apply(
        lambda row: f"FINDINGS: {row.get('findings', '')} IMPRESSION: {row.get('impression', '')}", 
        axis=1
    )
    
    # Create binary labels for diseases (simplified)
    for label in CHEXPERT_LABELS:
        df[label] = 0  # Initialize all as 0
    
    # Simple keyword-based labeling (for demonstration)
    # In practice, use CheXpert labeler or manual annotations
    keyword_mapping = {
        'No Finding': ['normal', 'clear', 'unremarkable'],
        'Cardiomegaly': ['cardiomegaly', 'enlarged heart'],
        'Edema': ['edema', 'pulmonary edema'],
        'Consolidation': ['consolidation', 'infiltrate'],
        'Pneumonia': ['pneumonia'],
        'Atelectasis': ['atelectasis'],
        'Pneumothorax': ['pneumothorax'],
        'Pleural Effusion': ['effusion', 'pleural effusion'],
    }
    
    for label, keywords in keyword_mapping.items():
        for keyword in keywords:
            df.loc[df['report'].str.lower().str.contains(keyword, na=False), label] = 1
    
    # Verify image paths exist
    df['image_path'] = df['image'].apply(lambda x: os.path.join(image_dir, x))
    df = df[df['image_path'].apply(os.path.exists)]
    
    print(f"✓ Processed {len(df)} samples")
    return df


class RadiologyDataset(Dataset):
    """
    PyTorch Dataset for Radiology Report Generation
    """
    
    def __init__(self, dataframe, tokenizer, transform=None, max_length=256):
        """
        Args:
            dataframe: Pandas DataFrame with columns ['image_path', 'report', disease labels]
            tokenizer: Text tokenizer for report encoding
            transform: Image transformations
            max_length: Maximum sequence length for reports
        """
        self.data = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.disease_labels = CHEXPERT_LABELS
        
        # Default transforms if none provided
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.Grayscale(num_output_channels=3),  # Convert to 3 channels
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Load image
        try:
            image = Image.open(row['image_path']).convert('RGB')
            if self.transform:
                image = self.transform(image)
        except Exception as e:
            # Return black image if loading fails
            print(f"Error loading image {row['image_path']}: {e}")
            image = torch.zeros(3, 224, 224)
        
        # Tokenize report
        report = str(row['report'])
        encoding = self.tokenizer(
            report,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Extract disease labels
        labels = torch.tensor([
            float(row.get(label, 0)) for label in self.disease_labels
        ], dtype=torch.float32)
        
        return {
            'image': image,
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': labels,
            'report': report
        }


def create_dataloaders(train_df, val_df, test_df, tokenizer, batch_size=8, num_workers=4):
    """
    Create train, validation, and test dataloaders
    
    Args:
        train_df, val_df, test_df: DataFrames for each split
        tokenizer: Text tokenizer
        batch_size: Batch size
        num_workers: Number of dataloader workers
    
    Returns:
        train_loader, val_loader, test_loader
    """
    
    # Training augmentations
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Validation/test transforms (no augmentation)
    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = RadiologyDataset(train_df, tokenizer, transform=train_transform)
    val_dataset = RadiologyDataset(val_df, tokenizer, transform=eval_transform)
    test_dataset = RadiologyDataset(test_df, tokenizer, transform=eval_transform)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"✓ Created dataloaders:")
    print(f"  Train: {len(train_loader)} batches ({len(train_dataset)} samples)")
    print(f"  Val:   {len(val_loader)} batches ({len(val_dataset)} samples)")
    print(f"  Test:  {len(test_loader)} batches ({len(test_dataset)} samples)")
    
    return train_loader, val_loader, test_loader


def load_iu_xray(data_path):
    """
    Load IU X-Ray dataset
    
    Args:
        data_path: Path to IU X-Ray dataset
    
    Returns:
        dataframe: Processed DataFrame
    """
    print(f"Loading IU X-Ray from {data_path}...")
    
    # Load projections and reports
    projections_path = os.path.join(data_path, 'indiana_projections.csv')
    reports_path = os.path.join(data_path, 'indiana_reports.csv')
    
    projections = pd.read_csv(projections_path)
    reports = pd.read_csv(reports_path)
    
    # Merge
    data = projections.merge(reports, on='uid', how='inner')
    
    print(f"✓ Loaded {len(data)} samples")
    return data


def load_mimic_cxr(data_path):
    """
    Load MIMIC-CXR dataset
    
    Args:
        data_path: Path to MIMIC-CXR dataset
    
    Returns:
        train_df, val_df, test_df: DataFrames for each split
    """
    print(f"Loading MIMIC-CXR from {data_path}...")
    
    train = pd.read_csv(os.path.join(data_path, 'mimic_cxr_aug_train.csv'))
    val = pd.read_csv(os.path.join(data_path, 'mimic_cxr_val.csv'))
    test = pd.read_csv(os.path.join(data_path, 'mimic_cxr_test.csv'))
    
    print(f"✓ Loaded:")
    print(f"  Train: {len(train)} samples")
    print(f"  Val:   {len(val)} samples")
    print(f"  Test:  {len(test)} samples")
    
    return train, val, test


if __name__ == "__main__":
    print("Data preprocessing utilities loaded successfully!")
    print(f"CheXpert labels: {len(CHEXPERT_LABELS)}")
    print(CHEXPERT_LABELS)