import collections
import unittest

import torch
import torch.cuda
from test_torch import AbstractTestCases
from torch.testing._internal.common_utils import TestCase, run_tests

class TestCudaComm(TestCase):
    def test_foreach_tensor_add_scalar(self):
        N = 20
        H = 200
        W = 200

        for dt in torch.testing.get_all_dtypes():
            if dt == torch.bool: 
                continue
            
            for d in torch.testing.get_all_device_types():
                tensors = []
                for _ in range(N):
                    tensors.append(torch.zeros(H, W, device=d, dtype=dt))

                res = torch._foreach_add(tensors, 1)
                
                for t in res: 
                    self.assertEqual(t, torch.ones(H, W, device=d, dtype=dt))

if __name__ == '__main__':
    run_tests()
