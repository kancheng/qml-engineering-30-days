"""Ensure diagnostics retain failures instead of reporting false success."""

import sys
import unittest

from check_environment import probe_passed, run_command


class EnvironmentTests(unittest.TestCase):
    def test_command_failure_keeps_stderr_and_return_code(self):
        result = run_command([sys.executable, "-c", "import sys; print('failure', file=sys.stderr); sys.exit(3)"])
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["returncode"], 3)
        self.assertIn("failure", result["stderr"])

    def test_missing_command_is_recorded(self):
        self.assertEqual(run_command(["/nonexistent/day06-executable"])["status"], "missing")

    def test_timeout_keeps_partial_output(self):
        result = run_command([sys.executable, "-u", "-c", "import time; print('started'); time.sleep(10)"], timeout=0.1)
        self.assertEqual(result["status"], "timeout")
        self.assertIn("started", result["stdout"])

    def test_success_requires_matching_target_payload_and_exit_code(self):
        process = {"status": "ok", "returncode": 0}
        payload = {"backend": "nvidia", "target": "nvidia", "passed": True, "checks": {"state": True}}
        self.assertTrue(probe_passed(process, payload, "nvidia"))
        self.assertFalse(probe_passed(process, None, "nvidia"))
        self.assertFalse(probe_passed(process, payload, "qpp-cpu"))
        self.assertFalse(probe_passed({"status": "failed", "returncode": 1}, payload, "nvidia"))
        self.assertFalse(probe_passed(process, {**payload, "checks": {"state": False}}, "nvidia"))


if __name__ == "__main__":
    unittest.main()
