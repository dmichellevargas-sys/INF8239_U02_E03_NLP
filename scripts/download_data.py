"""Descarga reproducible del corpus (commit fijado) y construcción de los CSV procesados.

Uso:  uv run python scripts/download_data.py
"""
from __future__ import annotations

import json
import sys

import pandas as pd
import requests

from inf8239_u02.config import CORPUS_COMMIT, RAW_BASE_URL, RAW_FILES, ROOT, settings
from inf8239_u02.data import sha256, standardize

RAW_DIR = ROOT / "data/raw"
MANIFEST = ROOT / "data/raw_manifest.json"  # versionado: permite verificar la descarga


def download(name: str, filename: str) -> dict:
    target = RAW_DIR / filename
    url = f"{RAW_BASE_URL}/{filename}"
    if not target.exists():
        response = requests.get(url, timeout=120)
        response.raise_for_status()
        target.write_bytes(response.content)
    return {"name": name, "file": f"data/raw/{filename}", "url": url, "sha256": sha256(target)}


def main() -> int:
    if settings.data_source != "url":
        print("DATA_SOURCE distinto de 'url': se usarán los CSV ya presentes en data/processed.")
        return 0
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    entries = [download(name, filename) for name, filename in RAW_FILES.items()]

    if MANIFEST.exists():  # verificación de integridad contra la descarga registrada
        expected = {e["file"]: e["sha256"] for e in json.loads(MANIFEST.read_text())["files"]}
        for entry in entries:
            if expected.get(entry["file"]) not in (None, entry["sha256"]):
                print(f"ERROR: el hash de {entry['file']} no coincide con el manifiesto",
                      file=sys.stderr)
                return 1

    raw = {e["name"]: pd.read_excel(ROOT / e["file"]) for e in entries}
    corpus = pd.concat([
        standardize(raw["train"], "train"),
        standardize(raw["development"], "development"),
    ], ignore_index=True)
    external = standardize(raw["external_v2"], "external_v2")

    settings.dataset_path.parent.mkdir(parents=True, exist_ok=True)
    corpus.to_csv(settings.dataset_path, index=False)
    external.to_csv(settings.external_path, index=False)

    manifest = {
        "corpus": "The Spanish Fake News Corpus",
        "license": "CC BY 4.0",
        "commit": CORPUS_COMMIT,
        "files": entries,
        "processed": {
            "corpus.csv": {"rows": len(corpus), "sha256": sha256(settings.dataset_path)},
            "external_v2.csv": {"rows": len(external), "sha256": sha256(settings.external_path)},
        },
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Corpus principal: {len(corpus)} filas -> {settings.dataset_path.relative_to(ROOT)}")
    print(f"Externo v2.0:     {len(external)} filas -> {settings.external_path.relative_to(ROOT)}")
    for e in entries:
        print(f"  {e['file']}  SHA-256 {e['sha256'][:16]}…")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
