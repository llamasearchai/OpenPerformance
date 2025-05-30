#!/usr/bin/env python3
"""
Memory Profiling Script
Comprehensive memory usage profiling for ML workloads across frameworks.
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
import gc
from collections import defaultdict

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
    from transformers import AutoModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    from memory_profiler import profile
    MEMORY_PROFILER_AVAILABLE = True
except ImportError:
    MEMORY_PROFILER_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MemoryProfiler:
    """Memory usage profiler for ML workloads."""
    
    def __init__(self, workload: str, model_name: str, framework: str, duration_minutes: int):
        self.workload = workload
        self.model_name = model_name
        self.framework = framework
        self.duration_minutes = duration_minutes
        self.memory_snapshots = []
        self.gpu_memory_snapshots = []
        self.monitoring_active = False
        self.monitor_thread = None
        
    def start_monitoring(self):
        """Start memory monitoring in a background thread."""
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitor_memory)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        logger.info("Memory monitoring started")
        
    def stop_monitoring(self):
        """Stop memory monitoring."""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("Memory monitoring stopped")
        
    def _monitor_memory(self):
        """Monitor memory usage continuously."""
        while self.monitoring_active:
            # System memory
            memory = psutil.virtual_memory()
            process = psutil.Process()
            
            snapshot = {
                "timestamp": time.time(),
                "system_total_gb": memory.total / (1024**3),
                "system_available_gb": memory.available / (1024**3),
                "system_used_gb": memory.used / (1024**3),
                "system_percent": memory.percent,
                "process_rss_gb": process.memory_info().rss / (1024**3),
                "process_vms_gb": process.memory_info().vms / (1024**3),
                "process_percent": process.memory_percent()
            }
            
            # GPU memory if available
            gpu_snapshot = self._get_gpu_memory_snapshot()
            if gpu_snapshot:
                snapshot.update(gpu_snapshot)
                
            self.memory_snapshots.append(snapshot)
            time.sleep(0.1)  # Sample every 100ms
            
    def _get_gpu_memory_snapshot(self):
        """Get GPU memory snapshot if available."""
        gpu_data = {}
        
        if self.framework == "pytorch" and TORCH_AVAILABLE:
            if torch.cuda.is_available():
                try:
                    device = torch.cuda.current_device()
                    gpu_data.update({
                        "gpu_allocated_gb": torch.cuda.memory_allocated(device) / (1024**3),
                        "gpu_reserved_gb": torch.cuda.memory_reserved(device) / (1024**3),
                        "gpu_max_allocated_gb": torch.cuda.max_memory_allocated(device) / (1024**3),
                        "gpu_max_reserved_gb": torch.cuda.max_memory_reserved(device) / (1024**3)
                    })
                    
                    # Memory stats if available
                    try:
                        stats = torch.cuda.memory_stats(device)
                        gpu_data.update({
                            "gpu_active_gb": stats.get("active_bytes.all.current", 0) / (1024**3),
                            "gpu_inactive_gb": stats.get("inactive_split_bytes.all.current", 0) / (1024**3),
                            "gpu_num_alloc_retries": stats.get("num_alloc_retries", 0),
                            "gpu_num_ooms": stats.get("num_ooms", 0)
                        })
                    except:
                        pass
                        
                except Exception as e:
                    logger.warning(f"Failed to get PyTorch GPU memory stats: {e}")
                    
        elif self.framework == "tensorflow" and TF_AVAILABLE:
            try:
                gpus = tf.config.list_physical_devices('GPU')
                if gpus:
                    # TensorFlow memory info is more limited
                    try:
                        mem_info = tf.config.experimental.get_memory_info('GPU:0')
                        gpu_data.update({
                            "gpu_current_gb": mem_info.get('current', 0) / (1024**3),
                            "gpu_peak_gb": mem_info.get('peak', 0) / (1024**3)
                        })
                    except:
                        pass
            except Exception as e:
                logger.warning(f"Failed to get TensorFlow GPU memory stats: {e}")
                
        return gpu_data
        
    def create_pytorch_model(self):
        """Create PyTorch model for memory profiling."""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch not available")
            
        if self.model_name == "resnet50":
            model = torch_models.resnet50(pretrained=False)
            input_shape = (32, 3, 224, 224)
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
        
    def create_tensorflow_model(self):
        """Create TensorFlow model for memory profiling."""
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow not available")
            
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
            
        return model, input_shape
        
    def profile_pytorch_training(self):
        """Profile PyTorch training memory usage."""
        model, input_shape = self.create_pytorch_model()
        
        # Move to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
        # Create dummy data
        if self.model_name == "resnet50":
            inputs = torch.randn(*input_shape).to(device)
            targets = torch.randint(0, 1000, (input_shape[0],)).to(device)
        else:
            inputs = torch.randint(0, 30000, input_shape).to(device)
            targets = torch.randint(0, 1000, (input_shape[0],)).to(device)
            
        optimizer = torch.optim.Adam(model.parameters())
        criterion = nn.CrossEntropyLoss()
        
        # Start memory monitoring
        self.start_monitoring()
        
        try:
            start_time = time.time()
            iterations = 0
            
            while time.time() - start_time < self.duration_minutes * 60:
                optimizer.zero_grad()
                
                if self.model_name == "resnet50":
                    outputs = model(inputs)
                else:
                    outputs = model(inputs)
                    if hasattr(outputs, 'last_hidden_state'):
                        outputs = outputs.last_hidden_state
                    if len(outputs.shape) > 2:
                        outputs = outputs.mean(dim=1)
                    outputs = torch.nn.functional.linear(outputs, torch.randn(outputs.shape[-1], 1000).to(device))
                
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                iterations += 1
                
                # Periodic garbage collection
                if iterations % 100 == 0:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        
                if iterations % 10 == 0:
                    logger.info(f"Training iteration {iterations}")
                    
        finally:
            self.stop_monitoring()
            
        return {"iterations": iterations, "workload": "training"}
        
    def profile_pytorch_inference(self):
        """Profile PyTorch inference memory usage."""
        model, input_shape = self.create_pytorch_model()
        
        # Move to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()
        
        # Create dummy data
        if self.model_name == "resnet50":
            inputs = torch.randn(*input_shape).to(device)
        else:
            inputs = torch.randint(0, 30000, input_shape).to(device)
            
        # Start memory monitoring
        self.start_monitoring()
        
        try:
            start_time = time.time()
            iterations = 0
            
            with torch.no_grad():
                while time.time() - start_time < self.duration_minutes * 60:
                    if self.model_name == "resnet50":
                        outputs = model(inputs)
                    else:
                        outputs = model(inputs)
                        if hasattr(outputs, 'last_hidden_state'):
                            outputs = outputs.last_hidden_state
                            
                    # Ensure computation is complete
                    if hasattr(outputs, 'cpu'):
                        _ = outputs.cpu()
                        
                    iterations += 1
                    
                    # Periodic garbage collection
                    if iterations % 1000 == 0:
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            
                    if iterations % 100 == 0:
                        logger.info(f"Inference iteration {iterations}")
                        
        finally:
            self.stop_monitoring()
            
        return {"iterations": iterations, "workload": "inference"}
        
    def profile_tensorflow_training(self):
        """Profile TensorFlow training memory usage."""
        model, input_shape = self.create_tensorflow_model()
        
        # Create dummy data
        if self.model_name == "resnet50":
            inputs = tf.random.normal(input_shape)
            targets = tf.random.uniform((input_shape[0],), 0, 1000, dtype=tf.int32)
        else:
            inputs = tf.random.uniform(input_shape, 0, 30000, dtype=tf.int32)
            targets = tf.random.uniform((input_shape[0],), 0, 1000, dtype=tf.int32)
            
        optimizer = tf.keras.optimizers.Adam()
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        
        # Start memory monitoring
        self.start_monitoring()
        
        try:
            start_time = time.time()
            iterations = 0
            
            while time.time() - start_time < self.duration_minutes * 60:
                with tf.GradientTape() as tape:
                    outputs = model(inputs, training=True)
                    if len(outputs.shape) > 2:
                        outputs = tf.reduce_mean(outputs, axis=1)
                    loss = loss_fn(targets, outputs)
                
                gradients = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                
                iterations += 1
                
                # Periodic garbage collection
                if iterations % 100 == 0:
                    gc.collect()
                    
                if iterations % 10 == 0:
                    logger.info(f"Training iteration {iterations}")
                    
        finally:
            self.stop_monitoring()
            
        return {"iterations": iterations, "workload": "training"}
        
    def profile_tensorflow_inference(self):
        """Profile TensorFlow inference memory usage."""
        model, input_shape = self.create_tensorflow_model()
        
        # Create dummy data
        if self.model_name == "resnet50":
            inputs = tf.random.normal(input_shape)
        else:
            inputs = tf.random.uniform(input_shape, 0, 30000, dtype=tf.int32)
            
        # Start memory monitoring
        self.start_monitoring()
        
        try:
            start_time = time.time()
            iterations = 0
            
            while time.time() - start_time < self.duration_minutes * 60:
                outputs = model(inputs, training=False)
                if len(outputs.shape) > 2:
                    outputs = tf.reduce_mean(outputs, axis=1)
                    
                # Ensure computation is complete
                _ = outputs.numpy()
                
                iterations += 1
                
                # Periodic garbage collection
                if iterations % 1000 == 0:
                    gc.collect()
                    
                if iterations % 100 == 0:
                    logger.info(f"Inference iteration {iterations}")
                    
        finally:
            self.stop_monitoring()
            
        return {"iterations": iterations, "workload": "inference"}
        
    def analyze_memory_usage(self):
        """Analyze collected memory usage data."""
        if not self.memory_snapshots:
            return {}
            
        # Basic statistics
        system_memory = [s["system_percent"] for s in self.memory_snapshots]
        process_memory = [s["process_rss_gb"] for s in self.memory_snapshots]
        
        analysis = {
            "duration_seconds": self.memory_snapshots[-1]["timestamp"] - self.memory_snapshots[0]["timestamp"],
            "sample_count": len(self.memory_snapshots),
            "system_memory": {
                "peak_percent": max(system_memory),
                "min_percent": min(system_memory),
                "avg_percent": sum(system_memory) / len(system_memory),
                "peak_used_gb": max(s["system_used_gb"] for s in self.memory_snapshots),
                "total_gb": self.memory_snapshots[0]["system_total_gb"]
            },
            "process_memory": {
                "peak_rss_gb": max(process_memory),
                "min_rss_gb": min(process_memory),
                "avg_rss_gb": sum(process_memory) / len(process_memory),
                "peak_percent": max(s["process_percent"] for s in self.memory_snapshots)
            }
        }
        
        # GPU analysis if available
        gpu_snapshots = [s for s in self.memory_snapshots if "gpu_allocated_gb" in s]
        if gpu_snapshots:
            gpu_allocated = [s["gpu_allocated_gb"] for s in gpu_snapshots]
            gpu_reserved = [s["gpu_reserved_gb"] for s in gpu_snapshots]
            
            analysis["gpu_memory"] = {
                "peak_allocated_gb": max(gpu_allocated),
                "avg_allocated_gb": sum(gpu_allocated) / len(gpu_allocated),
                "peak_reserved_gb": max(gpu_reserved),
                "avg_reserved_gb": sum(gpu_reserved) / len(gpu_reserved),
                "max_allocated_gb": max(s.get("gpu_max_allocated_gb", 0) for s in gpu_snapshots),
                "max_reserved_gb": max(s.get("gpu_max_reserved_gb", 0) for s in gpu_snapshots)
            }
            
            # Check for memory issues
            oom_count = max(s.get("gpu_num_ooms", 0) for s in gpu_snapshots)
            retry_count = max(s.get("gpu_num_alloc_retries", 0) for s in gpu_snapshots)
            
            analysis["gpu_memory"]["oom_count"] = oom_count
            analysis["gpu_memory"]["alloc_retry_count"] = retry_count
            
        # Memory growth analysis
        if len(process_memory) > 10:
            # Split into chunks to analyze growth
            chunk_size = len(process_memory) // 5
            chunks = [process_memory[i:i+chunk_size] for i in range(0, len(process_memory), chunk_size)]
            chunk_avgs = [sum(chunk) / len(chunk) for chunk in chunks if chunk]
            
            if len(chunk_avgs) >= 2:
                growth_rate = (chunk_avgs[-1] - chunk_avgs[0]) / len(chunk_avgs)
                analysis["memory_growth"] = {
                    "growth_rate_gb_per_interval": growth_rate,
                    "trend": "increasing" if growth_rate > 0.01 else "decreasing" if growth_rate < -0.01 else "stable"
                }
                
        # Memory efficiency
        if gpu_snapshots:
            # GPU utilization efficiency
            gpu_util = analysis["gpu_memory"]["peak_allocated_gb"] / analysis["gpu_memory"]["peak_reserved_gb"] if analysis["gpu_memory"]["peak_reserved_gb"] > 0 else 0
            analysis["memory_efficiency"] = {
                "gpu_utilization_ratio": gpu_util,
                "memory_fragmentation": 1 - gpu_util if gpu_util > 0 else 0
            }
            
        return analysis
        
    def run_profiling(self) -> Dict[str, Any]:
        """Run memory profiling for the specified workload."""
        logger.info(f"Starting memory profiling: {self.workload} with {self.model_name} on {self.framework}")
        logger.info(f"Duration: {self.duration_minutes} minutes")
        
        start_time = time.time()
        
        try:
            if self.framework == "pytorch":
                if not TORCH_AVAILABLE:
                    raise ImportError("PyTorch not available")
                    
                if self.workload == "training":
                    workload_results = self.profile_pytorch_training()
                elif self.workload == "inference":
                    workload_results = self.profile_pytorch_inference()
                else:
                    raise ValueError(f"Unknown workload: {self.workload}")
                    
            elif self.framework == "tensorflow":
                if not TF_AVAILABLE:
                    raise ImportError("TensorFlow not available")
                    
                if self.workload == "training":
                    workload_results = self.profile_tensorflow_training()
                elif self.workload == "inference":
                    workload_results = self.profile_tensorflow_inference()
                else:
                    raise ValueError(f"Unknown workload: {self.workload}")
                    
            else:
                raise ValueError(f"Unknown framework: {self.framework}")
                
            # Analyze memory usage
            memory_analysis = self.analyze_memory_usage()
            
            results = {
                "workload": self.workload,
                "model": self.model_name,
                "framework": self.framework,
                "duration_minutes": self.duration_minutes,
                "benchmark_type": "memory_profiling",
                "timestamp": time.time(),
                "success": True,
                "error": None,
                "workload_results": workload_results,
                "memory_analysis": memory_analysis,
                "raw_snapshots": self.memory_snapshots[-100:]  # Keep last 100 samples
            }
            
        except Exception as e:
            logger.error(f"Memory profiling failed: {e}")
            results = {
                "workload": self.workload,
                "model": self.model_name,
                "framework": self.framework,
                "duration_minutes": self.duration_minutes,
                "benchmark_type": "memory_profiling",
                "timestamp": time.time(),
                "success": False,
                "error": str(e),
                "total_time": time.time() - start_time,
                "memory_analysis": self.analyze_memory_usage() if self.memory_snapshots else {}
            }
            
        return results


def main():
    parser = argparse.ArgumentParser(description="Memory Profiling for ML Workloads")
    parser.add_argument("--workload", required=True,
                       choices=["training", "inference"],
                       help="Type of workload to profile")
    parser.add_argument("--model", required=True, 
                       choices=["resnet50", "bert-base"],
                       help="Model to profile")
    parser.add_argument("--framework", required=True,
                       choices=["pytorch", "tensorflow"],
                       help="Framework to use")
    parser.add_argument("--duration", type=int, default=5,
                       help="Duration in minutes")
    parser.add_argument("--output", required=True,
                       help="Output JSON file")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Run profiling
    profiler = MemoryProfiler(
        args.workload, args.model, args.framework, args.duration
    )
    
    results = profiler.run_profiling()
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
        
    logger.info(f"Memory profiling completed. Results saved to {args.output}")
    
    # Print summary
    if results.get("success") and "memory_analysis" in results:
        analysis = results["memory_analysis"]
        if "process_memory" in analysis:
            logger.info(f"Peak process memory: {analysis['process_memory']['peak_rss_gb']:.2f} GB")
        if "gpu_memory" in analysis:
            logger.info(f"Peak GPU memory: {analysis['gpu_memory']['peak_allocated_gb']:.2f} GB")


if __name__ == "__main__":
    main() 