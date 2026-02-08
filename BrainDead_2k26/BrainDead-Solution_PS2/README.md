
# BrainDead-Solution: Cognitive Radiology Report Generation

## Hackathon Submission - Medical AI Report Generator

A state-of-the-art deep learning framework for automated radiology report generation from chest X-ray images. This system combines hierarchical visual perception, knowledge-enhanced classification, and triangular cognitive attention to generate clinically accurate medical reports.

---

### Three Core Modules:

1. **PRO-FA (encoder.py)**: Hierarchical visual perception with multi-scale attention
2. **MIX-MLP (classifier.py)**: Knowledge-enhanced multi-label disease classification
3. **RCTA (decoder.py)**: Reinforced cross-modal triangular attention for report generation

---

## ✨ Key Features

- **Multi-Task Learning**: Simultaneous disease classification and report generation
- **Hierarchical Vision**: Progressive region-object focused attention
- **Cross-Modal Fusion**: Triangular attention between visual, semantic, and textual features
- **Medical Language Model**: Fine-tuned BioBERT for clinical report generation
- **14 Disease Classes**: CheXpert-compatible pathology detection
- **Production Ready**: Modular architecture with inference pipeline

---

## 🚀 Installation

### 1. Clone Repository

```bash
git clone https://github.com/yourusername/BrainDead-Solution.git
cd BrainDead-Solution
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
```

---

## 📊 Dataset Setup

### Supported Datasets:

1. **IU X-Ray**: Indiana University Chest X-Ray Collection
2. **MIMIC-CXR**: MIMIC Chest X-Ray Database

### Download Instructions:

#### IU X-Ray Dataset

```bash
# Download from PhysioNet
# https://physionet.org/content/indiana-chest-x-rays/

# Expected structure:
data/
  iu-xray/
    images/
      *.png
    indiana_projections.csv
    indiana_reports.csv
```

#### MIMIC-CXR Dataset

```bash
# Requires PhysioNet credentialed access
# https://physionet.org/content/mimic-cxr/

# Expected structure:
data/
  mimic-cxr/
    files/
      p10/
      p11/
      ...
    mimic_cxr_aug_train.csv
    mimic_cxr_val.csv
    mimic_cxr_test.csv
```

### Preprocess Data

```python
from data.preprocess import load_iu_xray, prepare_iu_dataset

# Load and prepare dataset
df = load_iu_xray('data/iu-xray')
processed_df = prepare_iu_dataset(df, 'data/iu-xray/images')
```

---

## 💻 Usage

### Training

```bash
python train.py --config config.yaml
```

**Configuration Example** (`config.yaml`):

```yaml
# Model
embed_dim: 512
num_classes: 14

# Training
batch_size: 8
num_epochs: 10
learning_rate: 2e-5
weight_decay: 0.01

# Data
data_path: 'data/iu-xray'
output_path: 'outputs'

# Loss weights
cls_weight: 0.3
gen_weight: 0.7
```

### Inference

```python
from models.encoder import HierarchicalVisualEncoder
from models.classifier import MixMLP
from models.decoder import ReportGenerationDecoder
import torch
from PIL import Image

# Load model
model = CognitiveRadiologyModel()
model.load_state_dict(torch.load('outputs/models/best_model.pt'))
model.eval()

# Load image
image = Image.open('sample_xray.png')
image_tensor = preprocess(image)

# Generate report
with torch.no_grad():
    report, diseases = model.generate_report(image_tensor)

print("Generated Report:", report)
print("Detected Diseases:", diseases)
```

### Using Inference Notebook

```bash
jupyter notebook notebooks/inference.ipynb
```

---

## 🧩 Model Components

### 1. PRO-FA Encoder (`models/encoder.py`)

**Progressive Region-Object Focused Attention**

- Multi-scale feature extraction (ResNet101 backbone)
- Hierarchical attention mechanisms
- Adaptive feature fusion

```python
from models.encoder import HierarchicalVisualEncoder

encoder = HierarchicalVisualEncoder(embed_dim=512)
visual_features, attention_maps = encoder(images)
```

### 2. MIX-MLP Classifier (`models/classifier.py`)

**Multi-path Information eXchange MLP**

- 3-path parallel processing
- Cross-path attention
- Ensemble prediction fusion

```python
from models.classifier import MixMLP

classifier = MixMLP(input_dim=512, num_classes=14)
disease_logits, path_logits = classifier(visual_features)
```

### 3. RCTA Decoder (`models/decoder.py`)

**Reinforced Cross-modal Triangular Attention**

- Triangular attention flow (Visual ↔ Semantic ↔ Textual)
- BioBERT language model integration
- Autoregressive report generation

```python
from models.decoder import ReportGenerationDecoder

decoder = ReportGenerationDecoder(embed_dim=512)
reports = decoder.generate(visual_features, semantic_features, tokenizer)
```

---

## 📈 Evaluation

### CheXpert Classification

```python
from evaluation.evaluate import CheXpertEvaluator

evaluator = CheXpertEvaluator(class_names=DISEASE_LABELS)
metrics, predictions, ground_truth = evaluator.evaluate(model, test_loader, device)
evaluator.print_results(metrics)
```

**Metrics:**
- Per-class F1 Score
- Precision & Recall
- AUC-ROC
- Macro-averaged metrics

### RadGraph Report Quality

```python
from evaluation.evaluate import RadGraphEvaluator

evaluator = RadGraphEvaluator(tokenizer)
metrics, generated, references = evaluator.evaluate(model, test_loader, device)
evaluator.print_results(metrics)
```

**Metrics:**
- Word-level Precision/Recall/F1
- BLEU Score (optional)
- Clinical entity matching

---

## Advanced Usage

### Custom Training Script

```python
from training.train import create_trainer
from data.preprocess import create_dataloaders

# Prepare data
train_loader, val_loader, test_loader = create_dataloaders(
    train_df, val_df, test_df, tokenizer, batch_size=8
)

# Create trainer
config = {
    'num_epochs': 10,
    'learning_rate': 2e-5,
    'output_path': 'outputs'
}
trainer = create_trainer(model, train_loader, val_loader, config)

# Train
history = trainer.train()
```

### Multi-GPU Training

```python
if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)
```

---

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@misc{braindead2024cognitive,
  title={Cognitive Radiology Report Generation: A Multi-Task Deep Learning Framework},
  author={BrainDead Team},
  year={2024},
  howpublished={Hackathon Submission}
}
```

---

## 👥 Team

**M0delN0tF0und**

---

