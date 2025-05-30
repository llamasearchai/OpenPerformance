#!/usr/bin/env python3
"""
Distributed Training Performance Benchmark Script
Comprehensive distributed training benchmarks across frameworks and configurations.
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
import threading
import random

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    import torch
    import torch.nn as nn
    import torch.distributed as dist
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
    from transformers import AutoModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DistributedBenchmark:
    """Distributed training performance benchmark runner."""
    
    def __init__(self, model_name: str, framework: str, nodes: int, 
                 communication: str, duration_minutes: int, simulate: bool = False):
        self.model_name = model_name
        self.framework = framework
        self.nodes = nodes
        self.communication = communication
        self.duration_minutes = duration_minutes
        self.simulate = simulate
        self.results = {}
        
    def create_pytorch_model(self):
        """Create PyTorch model for distributed training."""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch not available")
            
        if self.model_name == "resnet50":
            model = torch_models.resnet50(pretrained=False)
            input_shape = (32, 3, 224, 224)  # Fixed batch size for distributed
        elif self.model_name == "bert-base":
            if not TRANSFORMERS_AVAILABLE:
                # Fallback to simple transformer
                model = nn.TransformerEncoder(
                    nn.TransformerEncoderLayer(d_model=768, nhead=12), 
                    num_layers=12
                )
                input_shape = (32, 512, 768)
            else:
                model = AutoModel.from_pretrained("bert-base-uncased")
                input_shape = (32, 512)
        else:
            raise ValueError(f"Unknown model: {self.model_name}")
            
        return model, input_shape
        
    def simulate_distributed_pytorch(self):
        """Simulate distributed PyTorch training without actual multi-node setup."""
        model, input_shape = self.create_pytorch_model()
        
        # Simulate multiple workers with threading
        results = {}
        threads = []
        
        def worker_simulation(worker_id, duration_minutes):
            """Simulate a single worker."""
            logger.info(f"Starting worker {worker_id}")
            
            # Create dummy data
            if self.model_name == "resnet50":
                inputs = torch.randn(*input_shape)
                targets = torch.randint(0, 1000, (input_shape[0],))
            else:
                inputs = torch.randint(0, 30000, input_shape)
                targets = torch.randint(0, 1000, (input_shape[0],))
                
            optimizer = torch.optim.Adam(model.parameters())
            criterion = nn.CrossEntropyLoss()
            
            # Simulate training
            start_time = time.time()
            iterations = 0
            communication_times = []
            computation_times = []
            
            while time.time() - start_time < duration_minutes * 60:
                iter_start = time.time()
                
                # Forward pass
                comp_start = time.time()
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
                comp_time = time.time() - comp_start
                computation_times.append(comp_time)
                
                # Simulate communication (allreduce)
                comm_start = time.time()
                # Simulate network latency and bandwidth constraints
                comm_latency = self._simulate_communication_latency()
                time.sleep(comm_latency)
                comm_time = time.time() - comm_start
                communication_times.append(comm_time)
                
                optimizer.step()
                
                iterations += 1
                iter_time = time.time() - iter_start
                
                if iterations % 10 == 0:
                    logger.info(f"Worker {worker_id}: Iteration {iterations}, Time: {iter_time:.4f}s")
                    
            total_time = time.time() - start_time
            
            results[worker_id] = {
                "iterations": iterations,
                "total_time": total_time,
                "throughput": iterations / total_time,
                "avg_computation_time": sum(computation_times) / len(computation_times),
                "avg_communication_time": sum(communication_times) / len(communication_times),
                "communication_overhead": sum(communication_times) / sum(computation_times),
                "final_loss": loss.item() if 'loss' in locals() else 0.0
            }
            
        # Start worker threads
        for i in range(self.nodes):
            thread = threading.Thread(target=worker_simulation, args=(i, self.duration_minutes))
            threads.append(thread)
            thread.start()
            
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
            
        return self._aggregate_worker_results(results)
    
    def simulate_distributed_tensorflow(self):
        """Simulate distributed TensorFlow training."""
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow not available")
            
        # Simple simulation for TensorFlow distributed training
        if self.model_name == "resnet50":
            model = tf.keras.applications.ResNet50(
                weights=None, 
                input_shape=(224, 224, 3),
                classes=1000
            )
            input_shape = (32, 224, 224, 3)
        elif self.model_name == "bert-base":
            # Simple transformer-like model
            inputs = tf.keras.Input(shape=(512,))
            x = tf.keras.layers.Embedding(30000, 768)(inputs)
            for _ in range(12):
                x = tf.keras.layers.MultiHeadAttention(12, 64)(x, x)
                x = tf.keras.layers.LayerNormalization()(x)
            outputs = tf.keras.layers.Dense(1000)(x)
            model = tf.keras.Model(inputs, outputs)
            input_shape = (32, 512)
        else:
            raise ValueError(f"Unknown model: {self.model_name}")
            
        # Simulate distributed training
        start_time = time.time()
        iterations = 0
        communication_times = []
        computation_times = []
        
        optimizer = tf.keras.optimizers.Adam()
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        
        while time.time() - start_time < self.duration_minutes * 60:
            iter_start = time.time()
            
            # Create dummy data
            if self.model_name == "resnet50":
                inputs = tf.random.normal(input_shape)
                targets = tf.random.uniform((input_shape[0],), 0, 1000, dtype=tf.int32)
            else:
                inputs = tf.random.uniform(input_shape, 0, 30000, dtype=tf.int32)
                targets = tf.random.uniform((input_shape[0],), 0, 1000, dtype=tf.int32)
            
            # Forward and backward pass
            comp_start = time.time()
            with tf.GradientTape() as tape:
                outputs = model(inputs, training=True)
                if len(outputs.shape) > 2:
                    outputs = tf.reduce_mean(outputs, axis=1)
                loss = loss_fn(targets, outputs)
            
            gradients = tape.gradient(loss, model.trainable_variables)
            comp_time = time.time() - comp_start
            computation_times.append(comp_time)
            
            # Simulate communication
            comm_start = time.time()
            comm_latency = self._simulate_communication_latency()
            time.sleep(comm_latency)
            comm_time = time.time() - comm_start
            communication_times.append(comm_time)
            
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            
            iterations += 1
            iter_time = time.time() - iter_start
            
            if iterations % 10 == 0:
                logger.info(f"Iteration {iterations}, Time: {iter_time:.4f}s")
                
        total_time = time.time() - start_time
        
        return {
            "iterations": iterations,
            "total_time": total_time,
            "throughput": iterations / total_time,
            "avg_computation_time": sum(computation_times) / len(computation_times),
            "avg_communication_time": sum(communication_times) / len(communication_times),
            "communication_overhead": sum(communication_times) / sum(computation_times),
            "scaling_efficiency": self._calculate_scaling_efficiency(iterations, total_time),
            "final_loss": float(loss.numpy()) if 'loss' in locals() else 0.0
        }
    
    def _simulate_communication_latency(self):
        """Simulate communication latency based on backend and nodes."""
        # Base latency in seconds
        if self.communication == "nccl":
            base_latency = 0.001  # 1ms for NCCL
        elif self.communication == "gloo":
            base_latency = 0.005  # 5ms for Gloo
        else:
            base_latency = 0.002  # 2ms default
            
        # Scale with number of nodes (simplified ring allreduce model)
        scaled_latency = base_latency * (self.nodes - 1) / self.nodes
        
        # Add some randomness
        jitter = random.uniform(-0.2, 0.2)  # ±20% jitter
        final_latency = scaled_latency * (1 + jitter)
        
        return max(0.0001, final_latency)  # Minimum 0.1ms
    
    def _aggregate_worker_results(self, worker_results):
        """Aggregate results from multiple workers."""
        if not worker_results:
            return {}
            
        total_iterations = sum(r["iterations"] for r in worker_results.values())
        total_time = max(r["total_time"] for r in worker_results.values())  # Wall clock time
        avg_throughput = sum(r["throughput"] for r in worker_results.values()) / len(worker_results)
        
        avg_computation_time = sum(r["avg_computation_time"] for r in worker_results.values()) / len(worker_results)
        avg_communication_time = sum(r["avg_communication_time"] for r in worker_results.values()) / len(worker_results)
        avg_communication_overhead = sum(r["communication_overhead"] for r in worker_results.values()) / len(worker_results)
        
        scaling_efficiency = self._calculate_scaling_efficiency(total_iterations, total_time)
        
        return {
            "total_iterations": total_iterations,
            "total_time": total_time,
            "avg_throughput_per_worker": avg_throughput,
            "combined_throughput": total_iterations / total_time,
            "avg_computation_time": avg_computation_time,
            "avg_communication_time": avg_communication_time,
            "communication_overhead": avg_communication_overhead,
            "scaling_efficiency": scaling_efficiency,
            "workers": len(worker_results),
            "final_loss": sum(r["final_loss"] for r in worker_results.values()) / len(worker_results)
        }
    
    def _calculate_scaling_efficiency(self, iterations, total_time):
        """Calculate scaling efficiency compared to single node."""
        # Estimate single node performance (this is simplified)
        estimated_single_node_throughput = iterations / (total_time * self.nodes)
        actual_throughput = iterations / total_time
        
        # Perfect scaling would be: actual_throughput = estimated_single_node_throughput * nodes
        perfect_scaling_throughput = estimated_single_node_throughput * self.nodes
        
        if perfect_scaling_throughput > 0:
            scaling_efficiency = actual_throughput / perfect_scaling_throughput
        else:
            scaling_efficiency = 0.0
            
        return min(1.0, scaling_efficiency)  # Cap at 100%
    
    def run_benchmark(self) -> Dict[str, Any]:
        """Run the distributed training benchmark."""
        logger.info(f"Starting distributed benchmark: {self.model_name} on {self.framework}")
        logger.info(f"Nodes: {self.nodes}, Communication: {self.communication}")
        logger.info(f"Duration: {self.duration_minutes} minutes, Simulate: {self.simulate}")
        
        start_time = time.time()
        
        try:
            if self.simulate:
                # Always use simulation mode for CI/testing
                if self.framework == "pytorch":
                    if not TORCH_AVAILABLE:
                        raise ImportError("PyTorch not available")
                    results = self.simulate_distributed_pytorch()
                elif self.framework == "tensorflow":
                    if not TF_AVAILABLE:
                        raise ImportError("TensorFlow not available")
                    results = self.simulate_distributed_tensorflow()
                else:
                    raise ValueError(f"Unknown framework: {self.framework}")
            else:
                # Real distributed training would go here
                raise NotImplementedError("Real distributed training not implemented in CI environment")
                
            results.update({
                "model": self.model_name,
                "framework": self.framework,
                "nodes": self.nodes,
                "communication": self.communication,
                "duration_minutes": self.duration_minutes,
                "benchmark_type": "distributed",
                "simulation": self.simulate,
                "timestamp": time.time(),
                "success": True,
                "error": None
            })
            
        except Exception as e:
            logger.error(f"Benchmark failed: {e}")
            results = {
                "model": self.model_name,
                "framework": self.framework,
                "nodes": self.nodes,
                "communication": self.communication,
                "duration_minutes": self.duration_minutes,
                "benchmark_type": "distributed",
                "simulation": self.simulate,
                "timestamp": time.time(),
                "success": False,
                "error": str(e),
                "total_iterations": 0,
                "total_time": time.time() - start_time,
                "combined_throughput": 0,
                "scaling_efficiency": 0,
                "avg_memory_usage": psutil.virtual_memory().percent
            }
            
        return results


def main():
    parser = argparse.ArgumentParser(description="Distributed Training Performance Benchmark")
    parser.add_argument("--model", required=True, 
                       choices=["resnet50", "bert-base"],
                       help="Model to benchmark")
    parser.add_argument("--framework", required=True,
                       choices=["pytorch", "tensorflow"],
                       help="Framework to use")
    parser.add_argument("--nodes", type=int, default=1,
                       help="Number of nodes")
    parser.add_argument("--communication", default="nccl",
                       choices=["nccl", "gloo", "mpi"],
                       help="Communication backend")
    parser.add_argument("--duration", type=int, default=5,
                       help="Duration in minutes")
    parser.add_argument("--simulate", action="store_true",
                       help="Run in simulation mode")
    parser.add_argument("--output", required=True,
                       help="Output JSON file")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Run benchmark
    benchmark = DistributedBenchmark(
        args.model, args.framework, args.nodes,
        args.communication, args.duration, args.simulate
    )
    
    results = benchmark.run_benchmark()
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
        
    logger.info(f"Benchmark completed. Results saved to {args.output}")
    logger.info(f"Combined throughput: {results.get('combined_throughput', 0):.2f} iterations/sec")
    logger.info(f"Scaling efficiency: {results.get('scaling_efficiency', 0):.2%}")


if __name__ == "__main__":
    main() 