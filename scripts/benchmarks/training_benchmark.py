#!/usr/bin/env python3
"""
Training Performance Benchmark Script
Comprehensive training benchmarks for ML models across frameworks.
"""

import argparse
import json
import time
import psutil
import os
import sys
from pathlib import Path
from typing import Dict, Any, List
import logging

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    import torch
    import torch.nn as nn
    import torchvision.models as torch_models
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

try:
    from transformers import AutoModel, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TrainingBenchmark:
    """Training performance benchmark runner."""
    
    def __init__(self, model_name: str, framework: str, batch_size: int, 
                 precision: str, duration_minutes: int):
        self.model_name = model_name
        self.framework = framework
        self.batch_size = batch_size
        self.precision = precision
        self.duration_minutes = duration_minutes
        self.results = {}
        
    def create_pytorch_model(self):
        """Create PyTorch model."""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch not available")
            
        if self.model_name == "resnet50":
            model = torch_models.resnet50(pretrained=False)
            input_shape = (self.batch_size, 3, 224, 224)
        elif self.model_name == "bert-base":
            if not TRANSFORMERS_AVAILABLE:
                # Fallback to simple transformer
                model = nn.TransformerEncoder(
                    nn.TransformerEncoderLayer(d_model=768, nhead=12), 
                    num_layers=12
                )
                input_shape = (self.batch_size, 512, 768)
            else:
                model = AutoModel.from_pretrained("bert-base-uncased")
                input_shape = (self.batch_size, 512)
        elif self.model_name == "gpt2-small":
            if not TRANSFORMERS_AVAILABLE:
                # Fallback to simple transformer
                model = nn.TransformerDecoder(
                    nn.TransformerDecoderLayer(d_model=768, nhead=12),
                    num_layers=12
                )
                input_shape = (self.batch_size, 1024, 768)
            else:
                model = AutoModel.from_pretrained("gpt2")
                input_shape = (self.batch_size, 1024)
        else:
            raise ValueError(f"Unknown model: {self.model_name}")
            
        # Set precision
        if self.precision == "fp16":
            model = model.half()
            
        return model, input_shape
        
    def create_tensorflow_model(self):
        """Create TensorFlow model."""
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow not available")
            
        if self.model_name == "resnet50":
            model = tf.keras.applications.ResNet50(
                weights=None, 
                input_shape=(224, 224, 3),
                classes=1000
            )
            input_shape = (self.batch_size, 224, 224, 3)
        elif self.model_name == "bert-base":
            # Simple transformer-like model
            inputs = tf.keras.Input(shape=(512,))
            x = tf.keras.layers.Embedding(30000, 768)(inputs)
            for _ in range(12):
                x = tf.keras.layers.MultiHeadAttention(12, 64)(x, x)
                x = tf.keras.layers.LayerNormalization()(x)
            outputs = tf.keras.layers.Dense(1000)(x)
            model = tf.keras.Model(inputs, outputs)
            input_shape = (self.batch_size, 512)
        elif self.model_name == "gpt2-small":
            # Simple decoder-like model
            inputs = tf.keras.Input(shape=(1024,))
            x = tf.keras.layers.Embedding(50000, 768)(inputs)
            for _ in range(12):
                x = tf.keras.layers.MultiHeadAttention(12, 64)(x, x)
                x = tf.keras.layers.LayerNormalization()(x)
            outputs = tf.keras.layers.Dense(50000)(x)
            model = tf.keras.Model(inputs, outputs)
            input_shape = (self.batch_size, 1024)
        else:
            raise ValueError(f"Unknown model: {self.model_name}")
            
        return model, input_shape
        
    def benchmark_pytorch_training(self):
        """Benchmark PyTorch training."""
        model, input_shape = self.create_pytorch_model()
        
        # Create dummy data
        if self.model_name == "resnet50":
            inputs = torch.randn(*input_shape)
            targets = torch.randint(0, 1000, (self.batch_size,))
        else:
            inputs = torch.randint(0, 30000, input_shape)
            targets = torch.randint(0, 1000, (self.batch_size,))
            
        if self.precision == "fp16":
            inputs = inputs.half()
            
        optimizer = torch.optim.Adam(model.parameters())
        criterion = nn.CrossEntropyLoss()
        
        # Warmup
        for _ in range(10):
            optimizer.zero_grad()
            if self.model_name == "resnet50":
                outputs = model(inputs)
            else:
                outputs = model(inputs).last_hidden_state if hasattr(model(inputs), 'last_hidden_state') else model(inputs)
                if len(outputs.shape) > 2:
                    outputs = outputs.mean(dim=1)
                outputs = torch.nn.functional.linear(outputs, torch.randn(outputs.shape[-1], 1000))
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
        # Benchmark
        start_time = time.time()
        iterations = 0
        memory_usage = []
        
        while time.time() - start_time < self.duration_minutes * 60:
            iter_start = time.time()
            
            optimizer.zero_grad()
            if self.model_name == "resnet50":
                outputs = model(inputs)
            else:
                outputs = model(inputs).last_hidden_state if hasattr(model(inputs), 'last_hidden_state') else model(inputs)
                if len(outputs.shape) > 2:
                    outputs = outputs.mean(dim=1)
                outputs = torch.nn.functional.linear(outputs, torch.randn(outputs.shape[-1], 1000))
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            iterations += 1
            iter_time = time.time() - iter_start
            memory_usage.append(psutil.virtual_memory().percent)
            
            if iterations % 10 == 0:
                logger.info(f"Iteration {iterations}, Time: {iter_time:.4f}s, Loss: {loss.item():.4f}")
                
        total_time = time.time() - start_time
        throughput = iterations / total_time
        avg_memory = sum(memory_usage) / len(memory_usage)
        
        return {
            "iterations": iterations,
            "total_time": total_time,
            "throughput": throughput,
            "avg_iteration_time": total_time / iterations,
            "avg_memory_usage": avg_memory,
            "final_loss": loss.item()
        }
        
    def benchmark_tensorflow_training(self):
        """Benchmark TensorFlow training."""
        model, input_shape = self.create_tensorflow_model()
        
        # Create dummy data
        if self.model_name == "resnet50":
            inputs = tf.random.normal(input_shape)
            targets = tf.random.uniform((self.batch_size,), 0, 1000, dtype=tf.int32)
        else:
            inputs = tf.random.uniform(input_shape, 0, 30000, dtype=tf.int32)
            targets = tf.random.uniform((self.batch_size,), 0, 1000, dtype=tf.int32)
            
        optimizer = tf.keras.optimizers.Adam()
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        
        # Warmup
        for _ in range(10):
            with tf.GradientTape() as tape:
                outputs = model(inputs, training=True)
                if len(outputs.shape) > 2:
                    outputs = tf.reduce_mean(outputs, axis=1)
                loss = loss_fn(targets, outputs)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            
        # Benchmark
        start_time = time.time()
        iterations = 0
        memory_usage = []
        
        while time.time() - start_time < self.duration_minutes * 60:
            iter_start = time.time()
            
            with tf.GradientTape() as tape:
                outputs = model(inputs, training=True)
                if len(outputs.shape) > 2:
                    outputs = tf.reduce_mean(outputs, axis=1)
                loss = loss_fn(targets, outputs)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            
            iterations += 1
            iter_time = time.time() - iter_start
            memory_usage.append(psutil.virtual_memory().percent)
            
            if iterations % 10 == 0:
                logger.info(f"Iteration {iterations}, Time: {iter_time:.4f}s, Loss: {loss.numpy():.4f}")
                
        total_time = time.time() - start_time
        throughput = iterations / total_time
        avg_memory = sum(memory_usage) / len(memory_usage)
        
        return {
            "iterations": iterations,
            "total_time": total_time,
            "throughput": throughput,
            "avg_iteration_time": total_time / iterations,
            "avg_memory_usage": avg_memory,
            "final_loss": float(loss.numpy())
        }
        
    def run_benchmark(self) -> Dict[str, Any]:
        """Run the training benchmark."""
        logger.info(f"Starting training benchmark: {self.model_name} on {self.framework}")
        logger.info(f"Batch size: {self.batch_size}, Precision: {self.precision}")
        logger.info(f"Duration: {self.duration_minutes} minutes")
        
        start_time = time.time()
        
        try:
            if self.framework == "pytorch":
                if not TORCH_AVAILABLE:
                    raise ImportError("PyTorch not available")
                results = self.benchmark_pytorch_training()
            elif self.framework == "tensorflow":
                if not TF_AVAILABLE:
                    raise ImportError("TensorFlow not available")
                results = self.benchmark_tensorflow_training()
            else:
                raise ValueError(f"Unknown framework: {self.framework}")
                
            results.update({
                "model": self.model_name,
                "framework": self.framework,
                "batch_size": self.batch_size,
                "precision": self.precision,
                "duration_minutes": self.duration_minutes,
                "benchmark_type": "training",
                "timestamp": time.time(),
                "success": True,
                "error": None
            })
            
        except Exception as e:
            logger.error(f"Benchmark failed: {e}")
            results = {
                "model": self.model_name,
                "framework": self.framework,
                "batch_size": self.batch_size,
                "precision": self.precision,
                "duration_minutes": self.duration_minutes,
                "benchmark_type": "training",
                "timestamp": time.time(),
                "success": False,
                "error": str(e),
                "iterations": 0,
                "total_time": time.time() - start_time,
                "throughput": 0,
                "avg_iteration_time": 0,
                "avg_memory_usage": psutil.virtual_memory().percent
            }
            
        return results


def main():
    parser = argparse.ArgumentParser(description="Training Performance Benchmark")
    parser.add_argument("--model", required=True, 
                       choices=["resnet50", "bert-base", "gpt2-small"],
                       help="Model to benchmark")
    parser.add_argument("--framework", required=True,
                       choices=["pytorch", "tensorflow"],
                       help="Framework to use")
    parser.add_argument("--batch-size", type=int, default=32,
                       help="Batch size for training")
    parser.add_argument("--precision", default="fp32",
                       choices=["fp32", "fp16"],
                       help="Precision for training")
    parser.add_argument("--duration", type=int, default=5,
                       help="Duration in minutes")
    parser.add_argument("--output", required=True,
                       help="Output JSON file")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Run benchmark
    benchmark = TrainingBenchmark(
        args.model, args.framework, args.batch_size,
        args.precision, args.duration
    )
    
    results = benchmark.run_benchmark()
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
        
    logger.info(f"Benchmark completed. Results saved to {args.output}")
    logger.info(f"Throughput: {results.get('throughput', 0):.2f} iterations/sec")


if __name__ == "__main__":
    main() 