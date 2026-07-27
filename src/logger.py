"""Logger ghi log ra file + console, kèm save/load training state.

Giữ nguyên interface của class ``Logger`` trong notebooks để code cũ
chuyển sang dùng chung không phải đổi gì.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, Optional


class Logger:
    """Ghi log kèm elapsed time; state JSON đặt cạnh file log."""

    def __init__(self, log_path: str) -> None:
        self.log_path = log_path
        self.start = time.time()
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(
                f'\n===== Session started: '
                f'{time.strftime("%Y-%m-%d %H:%M:%S")} =====\n'
            )

    def log(self, msg: str, also_print: bool = True) -> None:
        """Ghi một dòng log (kèm elapsed seconds) ra file, và console nếu cần."""
        elapsed = time.time() - self.start
        line = f'[{elapsed:>8.1f}s] {msg}'
        with open(self.log_path, 'a', encoding='utf-8') as f:
            f.write(line + '\n')
        if also_print:
            print(line)

    def save_state(self, state: Dict[str, Any], name: str) -> str:
        """Lưu training state (step, epoch, loss, ...) ra ``<name>.json``."""
        path = os.path.join(os.path.dirname(self.log_path), f'{name}.json')
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
        self.log(f'State saved: {path}')
        return path

    def load_state(self, name: str) -> Optional[Dict[str, Any]]:
        """Load state đã lưu; trả ``None`` nếu chưa có (chạy lần đầu)."""
        path = os.path.join(os.path.dirname(self.log_path), f'{name}.json')
        if os.path.exists(path):
            with open(path, encoding='utf-8') as f:
                state = json.load(f)
            self.log(f'State loaded: {path}')
            return state
        return None
