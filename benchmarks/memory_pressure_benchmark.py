import sys
import os
import unittest
from unittest.mock import patch, MagicMock
import urllib.request
import json

# Ensure the parent directory is in sys.path to import shared.memory_pressure
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared import memory_pressure

class TestMemoryPressureScaling(unittest.TestCase):
    
    def setUp(self):
        self.base_budgets = {
            "system": 1000,
            "facts": 400,
            "history": 400,
            "tools": 300,
            "scratch": 80,
            "tiny": 10
        }
        
    @patch('shared.memory_pressure.psutil.virtual_memory')
    def test_relaxed_pressure(self, mock_vm):
        # 45% RAM
        mock_mem = MagicMock()
        mock_mem.percent = 45.0
        mock_vm.return_value = mock_mem
        
        scaled = memory_pressure.scale_budgets(self.base_budgets)
        self.assertEqual(scaled["system"], 1000)
        self.assertEqual(scaled["facts"], 400)
        self.assertEqual(scaled["tiny"], 20) # 10 * 1.0 = 10, floored to 20
        
    @patch('shared.memory_pressure.psutil.virtual_memory')
    def test_moderate_pressure(self, mock_vm):
        # 70% RAM
        mock_mem = MagicMock()
        mock_mem.percent = 70.0
        mock_vm.return_value = mock_mem
        
        scaled = memory_pressure.scale_budgets(self.base_budgets)
        self.assertEqual(scaled["system"], 1000)
        self.assertEqual(scaled["facts"], int(400 * 0.7))
        self.assertEqual(scaled["tiny"], 20)

    @patch('shared.memory_pressure.psutil.virtual_memory')
    def test_high_pressure(self, mock_vm):
        # 82% RAM
        mock_mem = MagicMock()
        mock_mem.percent = 82.0
        mock_vm.return_value = mock_mem
        
        scaled = memory_pressure.scale_budgets(self.base_budgets)
        self.assertEqual(scaled["system"], 1000)
        self.assertEqual(scaled["facts"], int(400 * 0.5))
        
    @patch('shared.memory_pressure.psutil.virtual_memory')
    def test_critical_pressure(self, mock_vm):
        # 90% RAM
        mock_mem = MagicMock()
        mock_mem.percent = 90.0
        mock_vm.return_value = mock_mem
        
        scaled = memory_pressure.scale_budgets(self.base_budgets)
        self.assertEqual(scaled["system"], 1000)
        self.assertEqual(scaled["facts"], int(400 * 0.3))

def run_api_tests():
    print("\n--- 2. API Test ---")
    try:
        req = urllib.request.Request("http://localhost:8000/api/system/pressure")
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode())
            print(f"GET /api/system/pressure returned: {data}")
            
            assert "tier" in data, "Missing 'tier'"
            assert "ram_percent" in data, "Missing 'ram_percent'"
            assert "scale_factor" in data, "Missing 'scale_factor'"
            assert "budgets_scaled" in data, "Missing 'budgets_scaled'"
            
            assert 0 <= data["ram_percent"] <= 100, "ram_percent out of bounds"
            assert data["tier"] in ["relaxed", "moderate", "high", "critical"], "Invalid tier"
            print("[PASS] API /api/system/pressure format is correct.")
    except Exception as e:
        print(f"[FAIL] API /api/system/pressure test failed: {e}")
        return False
        
    try:
        req = urllib.request.Request("http://localhost:8000/api/telemetry")
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode())
            assert "memory_pressure" in data, "Missing 'memory_pressure' in telemetry"
            print("[PASS] API /api/telemetry contains memory_pressure field.")
    except Exception as e:
        print(f"[FAIL] API /api/telemetry test failed: {e}")
        return False
        
    return True

def run_integration_test():
    print("\n--- 3. Integration Test ---")
    try:
        req = urllib.request.Request("http://localhost:8000/api/telemetry")
        with urllib.request.urlopen(req) as response:
            telemetry_data = json.loads(response.read().decode())
            telemetry_pressure = telemetry_data.get("memory_pressure", {})
            scale_factor = telemetry_pressure.get("scale_factor")
            tier = telemetry_pressure.get("tier")
            ram_percent = telemetry_pressure.get("ram_percent")
            
        req2 = urllib.request.Request("http://localhost:8000/api/system/pressure")
        with urllib.request.urlopen(req2) as response:
            sys_pressure_data = json.loads(response.read().decode())
            sys_scale = sys_pressure_data.get("scale_factor")
            
        assert scale_factor == sys_scale, "Scale factor mismatch between telemetry and system pressure"
        print("[PASS] Integration test: scale factors match.")
        print(f"Current RAM%: {ram_percent}%")
        print(f"Tier: {tier}")
        print(f"Scale Factor: {scale_factor}")
        
        # We don't have direct access to cache budgets via API, but we can print what they should be
        # based on base budgets in server.py
        base_budgets = {"facts": 400, "history": 400, "tools": 300, "scratch": 80}
        active_budgets = {k: max(20, int(v * scale_factor)) for k, v in base_budgets.items()}
        print(f"Expected Active Cache Budgets (excluding system): {active_budgets}")
        
    except Exception as e:
        print(f"[FAIL] Integration test failed: {e}")
        return False
    return True

if __name__ == "__main__":
    print("--- 1. Unit Tests ---")
    suite = unittest.TestLoader().loadTestsFromTestCase(TestMemoryPressureScaling)
    result = unittest.TextTestRunner(stream=sys.stdout, verbosity=2).run(suite)
    
    if not result.wasSuccessful():
        print("\n[FAIL] Unit tests failed.")
        sys.exit(1)
    else:
        print("\n[PASS] Unit tests passed.")
        
    api_passed = run_api_tests()
    int_passed = run_integration_test()
    
    print("\n--- Final Summary ---")
    if api_passed and int_passed:
        print("ALL TESTS PASSED SUCCESSFULLY.")
    else:
        print("SOME TESTS FAILED.")
