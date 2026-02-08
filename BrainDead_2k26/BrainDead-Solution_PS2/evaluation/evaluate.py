"""
CheXpert and RadGraph Evaluators
Evaluation metrics for medical report generation
"""

import torch
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.metrics import classification_report
import pandas as pd
from tqdm import tqdm


class CheXpertEvaluator:
    """
    CheXpert Classifier Evaluator
    Evaluates multi-label disease classification performance
    """
    
    def __init__(self, class_names):
        self.class_names = class_names
        self.num_classes = len(class_names)
    
    def evaluate(self, model, dataloader, device):
        """
        Evaluate classification performance
        
        Args:
            model: Trained model
            dataloader: Test data loader
            device: Device to run evaluation
        
        Returns:
            metrics: Dictionary with all metrics
            predictions: All predictions
            ground_truth: All ground truth labels
        """
        model.eval()
        
        all_predictions = []
        all_labels = []
        all_probs = []
        
        print("Running CheXpert Evaluation...")
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                images = batch['image'].to(device)
                labels = batch['labels'].to(device)
                
                # Get predictions
                if hasattr(model, 'classify'):
                    cls_logits = model.classify(images)
                else:
                    # If model returns both cls and gen
                    cls_logits, _ = model(images, batch['input_ids'].to(device))
                
                # Get probabilities
                probs = torch.sigmoid(cls_logits)
                
                # Convert to binary predictions (threshold = 0.5)
                preds = (probs > 0.5).float()
                
                all_predictions.append(preds.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
                all_probs.append(probs.cpu().numpy())
        
        # Concatenate all batches
        predictions = np.concatenate(all_predictions, axis=0)
        ground_truth = np.concatenate(all_labels, axis=0)
        probabilities = np.concatenate(all_probs, axis=0)
        
        # Compute metrics
        metrics = self._compute_metrics(predictions, ground_truth, probabilities)
        
        return metrics, predictions, ground_truth
    
    def _compute_metrics(self, predictions, ground_truth, probabilities):
        """Compute all classification metrics"""
        
        # Overall metrics
        overall_f1 = f1_score(ground_truth, predictions, average='macro')
        overall_precision = precision_score(ground_truth, predictions, average='macro', zero_division=0)
        overall_recall = recall_score(ground_truth, predictions, average='macro', zero_division=0)
        
        # Try to compute AUC (may fail if some classes have no positive samples)
        try:
            overall_auc = roc_auc_score(ground_truth, probabilities, average='macro')
        except:
            overall_auc = 0.0
        
        # Per-class metrics
        per_class_metrics = []
        for i, class_name in enumerate(self.class_names):
            class_f1 = f1_score(ground_truth[:, i], predictions[:, i], zero_division=0)
            class_precision = precision_score(ground_truth[:, i], predictions[:, i], zero_division=0)
            class_recall = recall_score(ground_truth[:, i], predictions[:, i], zero_division=0)
            
            try:
                class_auc = roc_auc_score(ground_truth[:, i], probabilities[:, i])
            except:
                class_auc = 0.0
            
            per_class_metrics.append({
                'disease': class_name,
                'f1': class_f1,
                'precision': class_precision,
                'recall': class_recall,
                'auc': class_auc
            })
        
        metrics = {
            'overall': {
                'f1': overall_f1,
                'precision': overall_precision,
                'recall': overall_recall,
                'auc': overall_auc
            },
            'per_class': per_class_metrics
        }
        
        return metrics
    
    def print_results(self, metrics):
        """Print evaluation results"""
        print("\n" + "=" * 80)
        print("CHEXPERT EVALUATION RESULTS")
        print("=" * 80)
        
        # Overall metrics
        print("\nOverall Metrics:")
        print(f"  F1 Score:  {metrics['overall']['f1']:.4f}")
        print(f"  Precision: {metrics['overall']['precision']:.4f}")
        print(f"  Recall:    {metrics['overall']['recall']:.4f}")
        print(f"  AUC:       {metrics['overall']['auc']:.4f}")
        
        # Per-class metrics
        print("\nPer-Class Metrics:")
        df = pd.DataFrame(metrics['per_class'])
        print(df.to_string(index=False))
        
        print("=" * 80)


class RadGraphEvaluator:
    """
    RadGraph Evaluator
    Evaluates report generation quality using entity matching
    
    Note: This is a simplified version. Full RadGraph requires
    NER model and entity graph matching.
    """
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def evaluate(self, model, dataloader, device, max_samples=100):
        """
        Evaluate report generation
        
        Args:
            model: Trained model
            dataloader: Test data loader
            device: Device to run evaluation
            max_samples: Maximum number of samples to evaluate
        
        Returns:
            metrics: Dictionary with generation metrics
            generated_reports: List of generated reports
        """
        model.eval()
        
        generated_reports = []
        reference_reports = []
        
        print("Running RadGraph Evaluation...")
        sample_count = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Generating Reports"):
                if sample_count >= max_samples:
                    break
                
                images = batch['image'].to(device)
                
                # Generate reports
                if hasattr(model, 'generate'):
                    reports = model.generate(
                        images=images,
                        tokenizer=self.tokenizer,
                        max_length=128,
                        num_beams=4
                    )
                else:
                    # Fallback: use forward pass
                    input_ids = batch['input_ids'].to(device)
                    cls_logits, gen_logits = model(images, input_ids)
                    
                    # Decode generated logits
                    predicted_ids = torch.argmax(gen_logits, dim=-1)
                    reports = self.tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)
                
                # Get reference reports
                references = self.tokenizer.batch_decode(batch['input_ids'], skip_special_tokens=True)
                
                generated_reports.extend(reports)
                reference_reports.extend(references)
                
                sample_count += len(reports)
        
        # Compute metrics
        metrics = self._compute_metrics(generated_reports, reference_reports)
        
        return metrics, generated_reports, reference_reports
    
    def _compute_metrics(self, generated, references):
        """
        Compute generation metrics
        
        Note: This is simplified. Full implementation would use:
        - BLEU score
        - ROUGE score
        - METEOR score
        - RadGraph F1 (entity matching)
        """
        
        # Simple word overlap metrics
        total_precision = 0
        total_recall = 0
        total_f1 = 0
        
        for gen, ref in zip(generated, references):
            gen_words = set(gen.lower().split())
            ref_words = set(ref.lower().split())
            
            if len(gen_words) == 0 or len(ref_words) == 0:
                continue
            
            overlap = gen_words.intersection(ref_words)
            
            precision = len(overlap) / len(gen_words) if len(gen_words) > 0 else 0
            recall = len(overlap) / len(ref_words) if len(ref_words) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            total_precision += precision
            total_recall += recall
            total_f1 += f1
        
        n = len(generated)
        metrics = {
            'word_precision': total_precision / n if n > 0 else 0,
            'word_recall': total_recall / n if n > 0 else 0,
            'word_f1': total_f1 / n if n > 0 else 0,
            'num_samples': n
        }
        
        return metrics
    
    def print_results(self, metrics, generated_reports=None, reference_reports=None, num_examples=3):
        """Print evaluation results"""
        print("\n" + "=" * 80)
        print("RADGRAPH EVALUATION RESULTS")
        print("=" * 80)
        
        print("\nGeneration Metrics:")
        print(f"  Word Precision: {metrics['word_precision']:.4f}")
        print(f"  Word Recall:    {metrics['word_recall']:.4f}")
        print(f"  Word F1:        {metrics['word_f1']:.4f}")
        print(f"  Samples:        {metrics['num_samples']}")
        
        # Show examples
        if generated_reports and reference_reports:
            print(f"\nExample Generated Reports (first {num_examples}):")
            for i in range(min(num_examples, len(generated_reports))):
                print(f"\n--- Example {i+1} ---")
                print(f"Reference: {reference_reports[i][:200]}...")
                print(f"Generated: {generated_reports[i][:200]}...")
        
        print("=" * 80)


def evaluate_model(model, test_loader, tokenizer, class_names, device):
    """
    Complete model evaluation with both CheXpert and RadGraph
    """
    print("\n" + "=" * 80)
    print("STARTING COMPLETE MODEL EVALUATION")
    print("=" * 80)
    
    # CheXpert evaluation
    chexpert_eval = CheXpertEvaluator(class_names)
    chexpert_metrics, predictions, ground_truth = chexpert_eval.evaluate(
        model, test_loader, device
    )
    chexpert_eval.print_results(chexpert_metrics)
    
    # RadGraph evaluation
    radgraph_eval = RadGraphEvaluator(tokenizer)
    radgraph_metrics, generated, references = radgraph_eval.evaluate(
        model, test_loader, device, max_samples=100
    )
    radgraph_eval.print_results(radgraph_metrics, generated, references)
    
    return {
        'chexpert': chexpert_metrics,
        'radgraph': radgraph_metrics,
        'predictions': predictions,
        'ground_truth': ground_truth,
        'generated_reports': generated,
        'reference_reports': references
    }