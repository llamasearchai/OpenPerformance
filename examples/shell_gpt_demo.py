#!/usr/bin/env python3
"""
Example demonstrating Shell-GPT integration with OpenPerformance CLI.

This script shows how to use the AI-powered shell assistance features
for ML performance engineering tasks.
"""

import subprocess
import sys
from pathlib import Path

def run_command(cmd: str, description: str = ""):
    """Run a CLI command and display the output."""
    if description:
        print(f"\n{'='*60}")
        print(f"DEMO: {description}")
        print(f"{'='*60}")
    
    print(f"\n$ {cmd}")
    print("-" * 40)
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    
    return result.returncode


def main():
    """Run Shell-GPT integration demos."""
    print("OpenPerformance Shell-GPT Integration Demo")
    print("=" * 60)
    
    # Check if OpenPerformance CLI is available
    if run_command("mlperf --help", "Check CLI installation") != 0:
        print("\nError: OpenPerformance CLI not found. Please install it first.")
        sys.exit(1)
    
    # Demo 1: Get shell command suggestions
    print("\n\nDemo 1: Shell Command Assistance")
    run_command(
        'mlperf gpt "how to monitor GPU memory usage in real-time"',
        "Get command suggestion for GPU monitoring"
    )
    
    # Demo 2: Performance query
    print("\n\nDemo 2: Performance Engineering Query")
    run_command(
        'mlperf chat "what are the best practices for optimizing transformer training on A100 GPUs"',
        "Ask about optimization best practices"
    )
    
    # Demo 3: Generate optimization script
    print("\n\nDemo 3: Generate Code")
    run_command(
        'mlperf gpt "generate a Python script to profile PyTorch model memory usage"',
        "Generate profiling script"
    )
    
    # Demo 4: Analyze performance issue
    print("\n\nDemo 4: Performance Analysis")
    run_command(
        'mlperf chat "my model training is showing low GPU utilization (40%). What could be the bottlenecks?" --agent optimization',
        "Analyze performance bottleneck"
    )
    
    # Demo 5: Get benchmarking recommendations
    print("\n\nDemo 5: Benchmarking Recommendations")
    run_command(
        'mlperf chat "recommend benchmarks for comparing FSDP vs DDP for a 7B parameter model" --agent benchmark',
        "Get benchmark recommendations"
    )
    
    print("\n\n" + "="*60)
    print("Demo completed!")
    print("\nTo start interactive mode, run:")
    print("  mlperf gpt --interactive")
    print("\nOr for AI chat mode:")
    print("  mlperf chat")


if __name__ == "__main__":
    main()