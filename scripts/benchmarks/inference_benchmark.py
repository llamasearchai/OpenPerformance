#!/usr/bin/env python3
"""
Inference Performance Benchmark Script
Comprehensive inference benchmarks for ML models across frameworks.
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
import statistics

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


class InferenceBenchmark:
    """Inference performance benchmark runner."""
    
    def __init__(self, model_name: str, framework: str, batch_size: int, 
                 precision: str, samples: int, warmup: int):
        self.model_name = model_name
        self.framework = framework
        self.batch_size = batch_size
        self.precision = precision
        self.samples = samples
        self.warmup = warmup
        self.results = {}
        
    def create_pytorch_model(self):
        """Create PyTorch model for inference."""
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
            
        # Set to evaluation mode
        model.eval()
            
        return model, input_shape
        
    def create_tensorflow_model(self):
        """Create TensorFlow model for inference."""
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
        
    def benchmark_pytorch_inference(self):
        """Benchmark PyTorch inference."""
        model, input_shape = self.create_pytorch_model()
        
        # Create dummy data
        if self.model_name == "resnet50":
            inputs = torch.randn(*input_shape)
        else:
            inputs = torch.randint(0, 30000, input_shape)
            
        if self.precision == "fp16":
            inputs = inputs.half()
            
        # Warmup runs
        logger.info(f"Running {self.warmup} warmup iterations...")
        with torch.no_grad():
            for _ in range(self.warmup):
                if self.model_name == "resnet50":
                    _ = model(inputs)
                else:
                    outputs = model(inputs)
                    if hasattr(outputs, 'last_hidden_state'):
                        _ = outputs.last_hidden_state
                    else:
                        _ = outputs
                        
        # Benchmark runs
        logger.info(f"Running {self.samples} benchmark iterations...")
        latencies = []
        memory_usage = []
        
        with torch.no_grad():
            for i in range(self.samples):
                start_time = time.time()
                
                if self.model_name == "resnet50":
                    outputs = model(inputs)
                else:
                    outputs = model(inputs)
                    if hasattr(outputs, 'last_hidden_state'):
                        outputs = outputs.last_hidden_state
                
                # Ensure computation is complete
                if hasattr(outputs, 'cpu'):
                    _ = outputs.cpu()
                
                end_time = time.time()
                latency = (end_time - start_time) * 1000  # Convert to milliseconds
                latencies.append(latency)
                memory_usage.append(psutil.virtual_memory().percent)
                
                if (i + 1) % 100 == 0:
                    logger.info(f"Completed {i + 1}/{self.samples} iterations")
        
        # Calculate statistics
        throughput = self.samples / (sum(latencies) / 1000)  # samples per second
        avg_latency = statistics.mean(latencies)
        p50_latency = statistics.median(latencies)
        p95_latency = statistics.quantiles(latencies, n=20)[18]  # 95th percentile
        p99_latency = statistics.quantiles(latencies, n=100)[98]  # 99th percentile
        min_latency = min(latencies)
        max_latency = max(latencies)
        avg_memory = statistics.mean(memory_usage)
        
        return {
            "samples": self.samples,
            "throughput_samples_per_sec": throughput,
            "avg_latency_ms": avg_latency,
            "p50_latency_ms": p50_latency,
            "p95_latency_ms": p95_latency,
            "p99_latency_ms": p99_latency,
            "min_latency_ms": min_latency,
            "max_latency_ms": max_latency,
            "latency_std_ms": statistics.stdev(latencies) if len(latencies) > 1 else 0,
            "avg_memory_usage": avg_memory,
            "total_inference_time_ms": sum(latencies)
        }
        
    def benchmark_tensorflow_inference(self):
        """Benchmark TensorFlow inference."""
        model, input_shape = self.create_tensorflow_model()
        
        # Create dummy data
        if self.model_name == "resnet50":
            inputs = tf.random.normal(input_shape)
        else:
            inputs = tf.random.uniform(input_shape, 0, 30000, dtype=tf.int32)
            
        # Warmup runs
        logger.info(f"Running {self.warmup} warmup iterations...")
        for _ in range(self.warmup):
            outputs = model(inputs, training=False)
            if len(outputs.shape) > 2:
                _ = tf.reduce_mean(outputs, axis=1)
                
        # Benchmark runs
        logger.info(f"Running {self.samples} benchmark iterations...")
        latencies = []
        memory_usage = []
        
        for i in range(self.samples):
            start_time = time.time()
            
            outputs = model(inputs, training=False)
            if len(outputs.shape) > 2:
                outputs = tf.reduce_mean(outputs, axis=1)
            
            # Ensure computation is complete
            _ = outputs.numpy()
            
            end_time = time.time()
            latency = (end_time - start_time) * 1000  # Convert to milliseconds
            latencies.append(latency)
            memory_usage.append(psutil.virtual_memory().percent)
            
            if (i + 1) % 100 == 0:
                logger.info(f"Completed {i + 1}/{self.samples} iterations")
        
        # Calculate statistics
        throughput = self.samples / (sum(latencies) / 1000)  # samples per second
        avg_latency = statistics.mean(latencies)
        p50_latency = statistics.median(latencies)
        p95_latency = statistics.quantiles(latencies, n=20)[18]  # 95th percentile
        p99_latency = statistics.quantiles(latencies, n=100)[98]  # 99th percentile
        min_latency = min(latencies)
        max_latency = max(latencies)
        avg_memory = statistics.mean(memory_usage)
        
        return {
            "samples": self.samples,
            "throughput_samples_per_sec": throughput,
            "avg_latency_ms": avg_latency,
            "p50_latency_ms": p50_latency,
            "p95_latency_ms": p95_latency,
            "p99_latency_ms": p99_latency,
            "min_latency_ms": min_latency,
            "max_latency_ms": max_latency,
            "latency_std_ms": statistics.stdev(latencies) if len(latencies) > 1 else 0,
            "avg_memory_usage": avg_memory,
            "total_inference_time_ms": sum(latencies)
        }
        
    def run_benchmark(self) -> Dict[str, Any]:
        """Run the inference benchmark."""
        logger.info(f"Starting inference benchmark: {self.model_name} on {self.framework}")
        logger.info(f"Batch size: {self.batch_size}, Precision: {self.precision}")
        logger.info(f"Samples: {self.samples}, Warmup: {self.warmup}")
        
        start_time = time.time()
        
        try:
            if self.framework == "pytorch":
                if not TORCH_AVAILABLE:
                    raise ImportError("PyTorch not available")
                results = self.benchmark_pytorch_inference()
            elif self.framework == "tensorflow":
                if not TF_AVAILABLE:
                    raise ImportError("TensorFlow not available")
                results = self.benchmark_tensorflow_inference()
            else:
                raise ValueError(f"Unknown framework: {self.framework}")
                
            results.update({
                "model": self.model_name,
                "framework": self.framework,
                "batch_size": self.batch_size,
                "precision": self.precision,
                "samples": self.samples,
                "warmup": self.warmup,
                "benchmark_type": "inference",
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
                "samples": self.samples,
                "warmup": self.warmup,
                "benchmark_type": "inference",
                "timestamp": time.time(),
                "success": False,
                "error": str(e),
                "throughput_samples_per_sec": 0,
                "avg_latency_ms": 0,
                "total_inference_time_ms": time.time() - start_time,
                "avg_memory_usage": psutil.virtual_memory().percent
            }
            
        return results


def main():
    parser = argparse.ArgumentParser(description="Inference Performance Benchmark")
    parser.add_argument("--model", required=True, 
                       choices=["resnet50", "bert-base", "gpt2-small"],
                       help="Model to benchmark")
    parser.add_argument("--framework", required=True,
                       choices=["pytorch", "tensorflow"],
                       help="Framework to use")
    parser.add_argument("--batch-size", type=int, default=1,
                       help="Batch size for inference")
    parser.add_argument("--precision", default="fp32",
                       choices=["fp32", "fp16"],
                       help="Precision for inference")
    parser.add_argument("--samples", type=int, default=1000,
                       help="Number of inference samples")
    parser.add_argument("--warmup", type=int, default=100,
                       help="Number of warmup iterations")
    parser.add_argument("--output", required=True,
                       help="Output JSON file")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Run benchmark
    benchmark = InferenceBenchmark(
        args.model, args.framework, args.batch_size,
        args.precision, args.samples, args.warmup
    )
    
    results = benchmark.run_benchmark()
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
        
    logger.info(f"Benchmark completed. Results saved to {args.output}")
    logger.info(f"Throughput: {results.get('throughput_samples_per_sec', 0):.2f} samples/sec")
    logger.info(f"Average latency: {results.get('avg_latency_ms', 0):.2f} ms")


if __name__ == "__main__":
    main() 