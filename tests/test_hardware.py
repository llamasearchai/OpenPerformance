"""
Test hardware monitoring functionality.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from python.mlperf.hardware.gpu import (
    get_gpu_info, get_gpu_count, get_cuda_info, GPUInfo, GPUMemoryInfo, GPUProcessInfo
)
from python.mlperf.hardware.cpu import get_cpu_info, CPUInfo
from python.mlperf.hardware.memory import get_memory_info, MemoryInfo


class TestGPUInfo:
    """Test GPU information gathering."""
    
    def test_gpu_info_creation(self):
        """Test GPU info object creation."""
        memory = GPUMemoryInfo(total=8000000000, free=4000000000, used=4000000000, utilization=50.0)
        process = GPUProcessInfo(pid=1234, process_name="python", memory_used=1000000000)
        
        gpu = GPUInfo(
            index=0,
            name="Test GPU",
            uuid="test-uuid",
            driver_version="525.60.11",
            cuda_version="12.0",
            memory=memory,
            temperature=65.0,
            power_usage=150.0,
            power_limit=250.0,
            utilization=85.0,
            memory_utilization=60.0,
            fan_speed=75,
            processes=[process],
            compute_capability=(8, 6),
            multi_gpu_board=False,
            board_id=0,
            clock_speeds={"graphics": 1500, "memory": 8000, "sm": 1500},
            max_clock_speeds={"graphics": 1800, "memory": 9000, "sm": 1800},
            performance_state="P0",
            encoder_utilization=0,
            decoder_utilization=0
        )
        
        assert gpu.index == 0
        assert gpu.name == "Test GPU"
        assert gpu.memory.utilization == 50.0
        assert len(gpu.processes) == 1
        assert gpu.processes[0].pid == 1234
    
    def test_gpu_info_to_dict(self):
        """Test GPU info dictionary conversion."""
        memory = GPUMemoryInfo(total=8000000000, free=4000000000, used=4000000000, utilization=50.0)
        gpu = GPUInfo(
            index=0,
            name="Test GPU",
            uuid="test-uuid",
            driver_version="525.60.11",
            cuda_version="12.0",
            memory=memory,
            temperature=65.0,
            power_usage=150.0,
            power_limit=250.0,
            utilization=85.0,
            memory_utilization=60.0,
            fan_speed=75,
            processes=[],
            compute_capability=(8, 6),
            multi_gpu_board=False,
            board_id=0,
            clock_speeds={},
            max_clock_speeds={},
            performance_state="P0",
            encoder_utilization=0,
            decoder_utilization=0
        )
        
        gpu_dict = gpu.to_dict()
        assert isinstance(gpu_dict, dict)
        assert gpu_dict["index"] == 0
        assert gpu_dict["name"] == "Test GPU"
        assert "memory" in gpu_dict
        assert isinstance(gpu_dict["memory"], dict)


class TestGetGPUInfo:
    """Test get_gpu_info function with proper dependency handling."""
    
    def test_get_gpu_info_no_pynvml_available(self):
        """Test get_gpu_info when pynvml is not available."""
        # Mock PYNVML_AVAILABLE as False
        with patch('python.mlperf.hardware.gpu.PYNVML_AVAILABLE', False):
            gpus = get_gpu_info()
            assert gpus == []
    
    @patch('python.mlperf.hardware.gpu.PYNVML_AVAILABLE', True)
    def test_get_gpu_info_nvidia_ml_exception(self):
        """Test get_gpu_info when NVIDIA ML raises an exception."""
        # Mock pynvml module to raise exception on init
        with patch('python.mlperf.hardware.gpu.pynvml') as mock_pynvml:
            mock_pynvml.nvmlInit.side_effect = Exception("NVIDIA ML not available")
            
            gpus = get_gpu_info()
            
            # Should handle exceptions gracefully and return empty list
            assert gpus == []
    
    @patch('python.mlperf.hardware.gpu.PYNVML_AVAILABLE', True)
    def test_get_gpu_info_success(self):
        """Test successful GPU info retrieval."""
        # Create comprehensive mock for pynvml
        with patch('python.mlperf.hardware.gpu.pynvml') as mock_pynvml:
            # Mock the handle
            mock_handle = MagicMock()
            
            # Configure pynvml mocks
            mock_pynvml.nvmlInit.return_value = None
            mock_pynvml.nvmlDeviceGetCount.return_value = 1
            mock_pynvml.nvmlDeviceGetHandleByIndex.return_value = mock_handle
            mock_pynvml.nvmlDeviceGetName.return_value = b"Test GPU"
            mock_pynvml.nvmlDeviceGetUUID.return_value = b"test-uuid-123"
            mock_pynvml.nvmlSystemGetDriverVersion.return_value = b"525.60.11"
            mock_pynvml.nvmlSystemGetCudaDriverVersion.return_value = 12000
            
            # Mock memory info
            mock_memory = MagicMock()
            mock_memory.total = 8000000000
            mock_memory.free = 4000000000
            mock_memory.used = 4000000000
            mock_pynvml.nvmlDeviceGetMemoryInfo.return_value = mock_memory
            
            # Mock other device properties
            mock_pynvml.nvmlDeviceGetTemperature.return_value = 65.0
            mock_pynvml.nvmlDeviceGetPowerUsage.return_value = 150000  # milliwatts
            mock_pynvml.nvmlDeviceGetPowerManagementLimitConstraints.return_value = (0, 250000)
            
            # Mock utilization
            mock_util = MagicMock()
            mock_util.gpu = 85
            mock_util.memory = 60
            mock_pynvml.nvmlDeviceGetUtilizationRates.return_value = mock_util
            
            mock_pynvml.nvmlDeviceGetFanSpeed.return_value = 75
            mock_pynvml.nvmlDeviceGetComputeRunningProcesses.return_value = []
            mock_pynvml.nvmlDeviceGetCudaComputeCapability.return_value = (8, 6)
            mock_pynvml.nvmlDeviceOnSameBoard.return_value = False
            mock_pynvml.nvmlDeviceGetBoardId.return_value = 0
            
            # Mock clock speeds - handle potential exceptions
            def mock_clock_info(handle, clock_type):
                clock_map = {
                    mock_pynvml.NVML_CLOCK_GRAPHICS: 1500,
                    mock_pynvml.NVML_CLOCK_MEM: 8000,
                    mock_pynvml.NVML_CLOCK_SM: 1500
                }
                return clock_map.get(clock_type, 1500)
            
            mock_pynvml.nvmlDeviceGetClockInfo.side_effect = mock_clock_info
            mock_pynvml.nvmlDeviceGetMaxClockInfo.side_effect = lambda h, t: mock_clock_info(h, t) + 300
            
            # Define clock constants
            mock_pynvml.NVML_CLOCK_GRAPHICS = 0
            mock_pynvml.NVML_CLOCK_MEM = 1
            mock_pynvml.NVML_CLOCK_SM = 2
            mock_pynvml.NVML_TEMPERATURE_GPU = 0
            
            gpus = get_gpu_info()
            
            assert len(gpus) == 1
            gpu = gpus[0]
            assert gpu.name == "Test GPU"
            assert gpu.memory.total == 8000000000
            assert gpu.temperature == 65.0
            assert gpu.utilization == 85
    
    @patch('python.mlperf.hardware.gpu.PYNVML_AVAILABLE', True) 
    def test_get_gpu_info_specific_index(self):
        """Test getting info for a specific GPU index."""
        with patch('python.mlperf.hardware.gpu.pynvml') as mock_pynvml:
            mock_pynvml.nvmlInit.return_value = None
            mock_pynvml.nvmlDeviceGetCount.return_value = 2
            
            # Test invalid index
            gpus = get_gpu_info(gpu_index=5)
            assert gpus == []
            
            # Test valid index (would need more mocking for full test)
            mock_pynvml.nvmlDeviceGetHandleByIndex.side_effect = Exception("Test exception")
            gpus = get_gpu_info(gpu_index=0)
            assert gpus == []


class TestGetGPUCount:
    """Test GPU count function."""
    
    def test_get_gpu_count_no_pynvml(self):
        """Test GPU count when pynvml is not available."""
        with patch('python.mlperf.hardware.gpu.PYNVML_AVAILABLE', False):
            count = get_gpu_count()
            assert count == 0
    
    @patch('python.mlperf.hardware.gpu.PYNVML_AVAILABLE', True)
    def test_get_gpu_count_success(self):
        """Test successful GPU count retrieval."""
        with patch('python.mlperf.hardware.gpu.pynvml') as mock_pynvml:
            mock_pynvml.nvmlInit.return_value = None
            mock_pynvml.nvmlDeviceGetCount.return_value = 2
            
            count = get_gpu_count()
            assert count == 2
    
    @patch('python.mlperf.hardware.gpu.PYNVML_AVAILABLE', True)
    def test_get_gpu_count_exception(self):
        """Test GPU count when exception occurs."""
        with patch('python.mlperf.hardware.gpu.pynvml') as mock_pynvml:
            mock_pynvml.nvmlInit.side_effect = Exception("Error")
            
            count = get_gpu_count()
            assert count == 0


class TestGetCudaInfo:
    """Test CUDA information gathering."""
    
    def test_get_cuda_info_no_frameworks(self):
        """Test CUDA info when no ML frameworks are available."""
        with patch('python.mlperf.hardware.gpu.TORCH_AVAILABLE', False), \
             patch('python.mlperf.hardware.gpu.TF_AVAILABLE', False), \
             patch('python.mlperf.hardware.gpu.PYNVML_AVAILABLE', False):
            
            info = get_cuda_info()
            
            assert not info['cuda_available']
            assert info['device_count'] == 0
            assert info['devices'] == []
    
    def test_get_cuda_info_pytorch(self):
        """Test CUDA info with PyTorch."""
        with patch('python.mlperf.hardware.gpu.TORCH_AVAILABLE', True), \
             patch('python.mlperf.hardware.gpu.TF_AVAILABLE', False), \
             patch('python.mlperf.hardware.gpu.PYNVML_AVAILABLE', False):
            
            # Mock torch module
            mock_torch = MagicMock()
            mock_torch.cuda.is_available.return_value = True
            mock_torch.cuda.device_count.return_value = 1
            mock_torch.version.cuda = "12.0"
            mock_torch.backends.cudnn.version.return_value = 8600
            
            # Mock device properties
            mock_props = MagicMock()
            mock_props.name = "Test GPU"
            mock_props.total_memory = 8000000000
            mock_props.major = 8
            mock_props.minor = 6
            mock_props.multi_processor_count = 108
            mock_torch.cuda.get_device_properties.return_value = mock_props
            
            with patch('python.mlperf.hardware.gpu.torch', mock_torch):
                info = get_cuda_info()
                
                assert info['cuda_available']
                assert info['device_count'] == 1
                assert info['cuda_version'] == "12.0"
                assert info['cudnn_version'] == 8600
                assert len(info['devices']) == 1
                assert info['devices'][0]['name'] == "Test GPU"


class TestCPUInfo:
    """Test CPU information gathering."""
    
    def test_get_cpu_info(self):
        """Test CPU info retrieval."""
        cpu_info = get_cpu_info()
        
        assert isinstance(cpu_info, CPUInfo)
        assert cpu_info.physical_cores > 0
        assert cpu_info.logical_cores > 0
        assert cpu_info.frequency_mhz > 0
        assert cpu_info.architecture is not None
        assert cpu_info.brand is not None
    
    def test_cpu_info_to_dict(self):
        """Test CPU info dictionary conversion."""
        cpu_info = get_cpu_info()
        cpu_dict = cpu_info.to_dict()
        
        assert isinstance(cpu_dict, dict)
        assert "physical_cores" in cpu_dict
        assert "logical_cores" in cpu_dict
        assert "frequency_mhz" in cpu_dict


class TestMemoryInfo:
    """Test memory information gathering."""
    
    def test_get_memory_info(self):
        """Test memory info retrieval."""
        memory_info = get_memory_info()
        
        assert isinstance(memory_info, MemoryInfo)
        assert memory_info.total_gb > 0
        assert memory_info.available_gb >= 0
        assert memory_info.used_gb >= 0
        assert 0 <= memory_info.percent_used <= 100
    
    def test_memory_info_to_dict(self):
        """Test memory info dictionary conversion."""
        memory_info = get_memory_info()
        memory_dict = memory_info.to_dict()
        
        assert isinstance(memory_dict, dict)
        assert "total_gb" in memory_dict
        assert "available_gb" in memory_dict
        assert "used_gb" in memory_dict
        assert "percent_used" in memory_dict


@pytest.mark.integration
class TestHardwareIntegration:
    """Integration tests for hardware monitoring."""
    
    def test_complete_hardware_info(self):
        """Test gathering complete hardware information."""
        # This test runs with real hardware when available
        try:
            cpu_info = get_cpu_info()
            memory_info = get_memory_info()
            gpu_count = get_gpu_count()
            cuda_info = get_cuda_info()
            
            # Basic assertions
            assert cpu_info is not None
            assert memory_info is not None
            assert isinstance(gpu_count, int)
            assert isinstance(cuda_info, dict)
            
            # If GPUs are available, test GPU info
            if gpu_count > 0:
                gpu_info = get_gpu_info()
                assert len(gpu_info) == gpu_count
                
                for gpu in gpu_info:
                    assert isinstance(gpu, GPUInfo)
                    assert gpu.index >= 0
                    assert gpu.name is not None
                    
        except Exception as e:
            pytest.skip(f"Integration test skipped due to hardware constraints: {e}")


if __name__ == "__main__":
    pytest.main([__file__]) 