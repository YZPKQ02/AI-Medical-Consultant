from __future__ import annotations

import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from app.core.config import load_env_file


class ConfigTests(unittest.TestCase):
    def test_load_env_file_accepts_utf8_bom(self):
        with TemporaryDirectory() as directory:
            env_path = Path(directory) / ".env"
            env_path.write_text("AGENT_LLM_ENABLED=1\n", encoding="utf-8-sig")

            with patch.dict(os.environ, {}, clear=True):
                load_env_file(env_path)

                self.assertEqual(os.environ["AGENT_LLM_ENABLED"], "1")


if __name__ == "__main__":
    unittest.main()
