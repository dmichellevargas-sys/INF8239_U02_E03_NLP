"""Ejecuta el flujo completo: descarga -> auditoría -> entrenamiento y evaluación.

Uso:  uv run python scripts/run_all.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import audit_data, download_data, train_text


def main() -> int:
    code = download_data.main()
    if code:
        return code
    audit_data.main()
    train_text.main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
