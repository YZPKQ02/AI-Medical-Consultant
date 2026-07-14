from __future__ import annotations

import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from app.core.config import env_bool, get_settings, load_env_file


class ConfigTests(unittest.TestCase):
    def test_load_env_file_accepts_utf8_bom(self):
        with TemporaryDirectory() as directory:
            env_path = Path(directory) / ".env"
            env_path.write_text("AGENT_LLM_ENABLED=1\n", encoding="utf-8-sig")

            with patch.dict(os.environ, {}, clear=True):
                load_env_file(env_path)

            self.assertEqual(os.environ["AGENT_LLM_ENABLED"], "1")

    def test_runtime_requirement_flags_are_loaded(self):
        with patch.dict(
            os.environ,
            {
                "APP_ENVIRONMENT": "production",
                "REQUIRE_POSTGRES": "1",
                "REQUIRE_QWEN": "true",
                "QWEN_DEVICE": "cuda:0",
            },
            clear=True,
        ):
            loaded = get_settings()

        self.assertEqual(loaded.app_environment, "production")
        self.assertTrue(loaded.require_postgres)
        self.assertTrue(loaded.require_qwen)
        self.assertEqual(loaded.qwen_device, "cuda:0")

    def test_env_bool_uses_default_for_missing_values(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(env_bool("MISSING"))
            self.assertTrue(env_bool("MISSING", default=True))


if __name__ == "__main__":
    unittest.main()
