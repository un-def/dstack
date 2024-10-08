import pytest

from dstack._internal.utils.gpu import convert_gpu_name

TESTS = [
    ("NVIDIA GeForce RTX 4060 Ti", "RTX4060Ti"),
    ("NVIDIA GeForce RTX 4060", "RTX4060"),
    ("NVIDIA RTX 4000 Ada Generation", "RTX4000Ada"),
    ("NVIDIA L4", "L4"),
    ("NVIDIA GH200 120GB", "GH200"),
    ("NVIDIA A100-SXM4-80GB", "A100"),
    ("NVIDIA A10G", "A10G"),
    ("NVIDIA L40S", "L40S"),
    ("NVIDIA H100 NVL", "H100NVL"),
    ("NVIDIA H100 80GB HBM3", "H100"),
    ("Tesla T4", "T4"),
]


class TestConvertGpuName:
    @pytest.mark.parametrize("test_input,expected", TESTS)
    def test_convert_gpu_name(self, test_input, expected):
        assert convert_gpu_name(test_input) == expected
