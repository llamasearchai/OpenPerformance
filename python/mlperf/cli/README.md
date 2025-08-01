# OpenPerformance CLI with Shell-GPT Integration

The OpenPerformance CLI provides powerful AI-assisted capabilities for ML performance engineering through integration with Shell-GPT and specialized AI agents.

## Features

### 1. Shell-GPT Integration
- AI-powered shell command assistance
- Interactive chat with GPT-4 for performance engineering
- Code generation for optimization tasks
- Command execution with AI guidance

### 2. Specialized AI Agents
- **BenchmarkAgent**: ML benchmark selection and analysis
- **OptimizationAgent**: Performance optimization recommendations
- **PerformanceAnalysisAgent**: Detailed performance insights and reports

### 3. Core CLI Commands
- Hardware information and monitoring
- Performance benchmarking
- Workload profiling
- Configuration optimization

## Installation

```bash
# Install OpenPerformance with all dependencies
pip install -e .

# Ensure Shell-GPT is installed
pip install shell-gpt
```

## Configuration

Set your OpenAI API key:
```bash
export OPENAI_API_KEY="your-api-key-here"
```

Or create a `.env` file:
```
OPENAI_API_KEY=your-api-key-here
```

## Usage Examples

### Basic Commands

```bash
# Show system information
mlperf info

# Run benchmark
mlperf benchmark --framework pytorch --batch-size 64

# Profile a script
mlperf profile train.py --track-memory
```

### Shell-GPT Commands

```bash
# Get shell command suggestions
mlperf gpt "how to monitor GPU memory usage"

# Execute suggested commands (with confirmation)
mlperf gpt "find large model checkpoint files" --execute

# Interactive shell assistance
mlperf gpt --interactive
```

### AI Agent Chat

```bash
# Chat with AI agents about performance
mlperf chat "why is my model training slow?"

# Use specific agent
mlperf chat "analyze these benchmark results" --agent benchmark

# Interactive chat mode
mlperf chat
```

### Advanced Examples

```bash
# Generate optimization code
mlperf gpt "generate PyTorch code for gradient accumulation"

# Analyze performance bottlenecks
mlperf chat "GPU utilization is 40%, how to improve?" --agent optimization

# Get benchmark recommendations
mlperf chat "recommend benchmarks for transformer models" --agent benchmark

# Performance report analysis
mlperf chat "analyze performance trends from last week" --agent performance
```

## Interactive Modes

### Shell-GPT Interactive Mode
```bash
mlperf gpt --interactive

# In interactive mode:
openperf> !nvidia-smi  # Execute command
openperf> ?find model files  # Get suggestion without execution
openperf> perf optimize transformer training  # Performance query
```

### AI Chat Mode
```bash
mlperf chat

# In chat mode:
You: How can I reduce memory usage in distributed training?
AI Assistant: [Detailed response with recommendations...]
```

## Command Reference

### mlperf gpt
AI-powered shell assistance with GPT integration.

Options:
- `query`: Query for AI assistant
- `--execute, -e`: Execute suggested commands
- `--interactive, -i`: Start interactive mode

### mlperf chat
Chat with specialized ML performance AI agents.

Options:
- `question`: Question to ask AI agents
- `--agent, -a`: Specific agent (auto/benchmark/optimization/performance)

### mlperf benchmark
Run performance benchmarks.

Options:
- `--framework, -f`: ML framework (pytorch/tensorflow/jax)
- `--model-size, -s`: Model size in GB
- `--batch-size, -b`: Batch size
- `--iterations, -i`: Number of iterations
- `--distributed, -d`: Enable distributed training simulation

### mlperf profile
Profile a Python script for performance analysis.

Options:
- `script`: Python script to profile
- `--framework, -f`: ML framework
- `--output-dir, -o`: Output directory for results
- `--track-memory`: Enable memory tracking

### mlperf optimize
Optimize ML workload based on configuration.

Options:
- `config_file`: Configuration file (JSON/YAML)
- `--dry-run`: Show recommendations without applying

## Best Practices

1. **Use specific agents for targeted assistance**:
   - Benchmark agent for testing strategies
   - Optimization agent for performance tuning
   - Performance agent for analysis and reports

2. **Provide context in queries**:
   - Include hardware specs (GPU model, memory)
   - Specify framework (PyTorch, TensorFlow)
   - Mention workload characteristics (model size, batch size)

3. **Iterate on recommendations**:
   - Start with high-level queries
   - Drill down into specific issues
   - Validate suggestions with benchmarks

## Troubleshooting

### OpenAI API Key Not Found
```bash
export OPENAI_API_KEY="your-key"
# Or add to .env file
```

### Shell-GPT Not Initialized
```bash
pip install shell-gpt
```

### Agent Initialization Failed
Check logs and ensure all dependencies are installed:
```bash
pip install -r requirements.txt
```

## Examples Directory

See `/examples/shell_gpt_demo.py` for a complete demonstration of Shell-GPT integration features.

## Contributing

To add new AI capabilities:

1. Create new agent in `python/mlperf/agents/`
2. Register agent in `SwarmManager`
3. Add CLI commands as needed
4. Update documentation

## Support

For issues or questions:
- Check the documentation
- Use `mlperf chat` to ask AI agents
- Submit issues on GitHub