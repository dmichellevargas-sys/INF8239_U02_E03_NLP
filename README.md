# INF-8239 · Ejercicio 03 · Corpus público y clasificador de texto reproducible

Detección de noticias falsas en español con **The Spanish Fake News Corpus** (CC BY 4.0).
Proyecto individual de la Unidad 02 (Procesamiento de Lenguaje Natural).
Proyecto base del curso: Edwin Ramón José Nolasco.

> Ejercicio académico. El modelo reconoce patrones de estilo y vocabulario; **no verifica hechos**.

## Pregunta
¿Puede un clasificador basado solo en el texto distinguir noticias falsas de verdaderas en español,
y cuánto de ese desempeño se mantiene con medios y periodos no vistos?

- **Unidad de análisis:** una noticia (titular + cuerpo).
- **Objetivo:** `label` ∈ {`fake`, `true`}.
- **Métrica principal:** F1 macro. **Error más costoso:** falsa clasificada como verdadera.

## Requisitos
- Python 3.12 (lo instala `uv`) y [uv](https://docs.astral.sh/uv/) ≥ 0.5.
- Conexión a internet la primera vez (descarga ≈ 2 MB desde GitHub).
- CPU; no requiere GPU. Ejecución completa < 1 minuto en un portátil.

## Ejecución (uv, recomendada)
```bash
uv python install 3.12
uv sync                                   # crea .venv con versiones exactas de uv.lock
uv run pytest -q                          # 26 pruebas
uv run python scripts/run_all.py          # descarga -> auditoría -> entrenamiento y evaluación
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/E03_clasificador_texto.ipynb
uv run streamlit run app/streamlit_app.py # demostración opcional
```
Pasos individuales: `scripts/download_data.py`, `scripts/audit_data.py`, `scripts/train_text.py`.
La configuración por defecto está en `src/inf8239_u02/config.py`; `.env.example` documenta las variables
(copiarlo como `.env` es opcional).

## Ejecución alternativa (pip)
```bash
python -m venv .venv
.venv\Scripts\Activate.ps1          # Windows PowerShell  (macOS/Linux: source .venv/bin/activate)
python -m pip install -r requirements.txt
$env:PYTHONPATH="src;."             # macOS/Linux: export PYTHONPATH=src:.
python -m pytest -q
python scripts/run_all.py
```

## Estructura
```
app/streamlit_app.py        demostración local
data/raw_manifest.json      URLs, commit y SHA-256 de la descarga (versionado)
data/raw, data/processed    generados por download_data.py (no versionados)
docs/                       DATASET_CARD, candidatos, categorías de errores, uso de IA
notebooks/                  notebook ejecutado de inicio a fin
reports/                    métricas (CSV/JSON) y figuras
scripts/                    descarga, auditoría, entrenamiento, run_all, embeddings (LAB06)
src/inf8239_u02/            config, data, audit, modeling, evaluation
tests/                      26 pruebas unitarias
```

## Decisiones principales
| Decisión | Razón |
|---|---|
| Corpus fijado al commit `732f4870` y verificado con SHA-256 | Descarga idéntica en cualquier equipo |
| Solo se usa `text`; se excluyen `source`, `link`, `id` | 147/150 medios publican una sola clase; solo el medio da F1 ≈ 0.93 (fuga) |
| Partición oficial train/development (676/295) | Comparable con la literatura y fijada por los autores |
| Se eliminan 1 duplicado interno y 4 cruzados de la prueba | Evitar evaluar con textos ya vistos |
| Cifras → token `number` | La v1.0 trae `*NUMBER*` y la v2.0 dígitos |
| Selección por validación cruzada en train con la regla de un error estándar | La prueba se usa una sola vez; ante empate estadístico, el modelo más barato |
| Evaluación por grupos de medios y con la v2.0 | Medir dependencia del medio y deriva temporal |

## Resultados (semilla 42, `uv.lock`)
| Modelo | F1 CV train | F1 prueba [IC 95 %] | Ajuste (s)* | Tamaño (KB) |
|---|---|---|---|---|
| dummy (línea base) | 0.332 | 0.324 [0.298, 0.351] | 0.80 | 763 |
| tfidf_word_logreg **(seleccionado)** | 0.828 ± 0.028 | 0.793 [0.744, 0.835] | 0.87 | 1017 |
| tfidf_char_svm | 0.834 ± 0.041 | 0.803 [0.753, 0.848] | 2.92 | 1554 |

\*Tiempos dependientes del hardware (varían entre ejecuciones); mediana de 3 ajustes. El F1 en entrenamiento es 1.0 en ambos pipelines (memorizan train). McNemar entre pipelines: p = 0.72.

| Escenario (modelo seleccionado) | F1 macro |
|---|---|
| Prueba oficial (medios vistos, 2018) | 0.793 |
| Medios no vistos (5 pliegues por grupo) | 0.703 ± 0.085 |
| Corpus v2.0 (2020-2021) | 0.673 [0.631, 0.710]; sin unificar cifras: 0.728 |

Análisis de 20 errores (10 FP y 10 FN más confiados): `reports/error_analysis_20.csv` y `fig_error_categories.png`.

## Artefactos
`reports/text_metrics.csv`, `cv_results.csv`, `text_results.json`, `audit_summary.json`,
`audit_source_purity.csv`, `robustness_unseen_sources.csv`, `errors_all.csv`, `error_analysis_20.csv`,
`top_terms.csv` y figuras `fig_*.png`. Los modelos (`models/*.joblib`) se regeneran y no se versionan.

## Git
```bash
git init
git add pyproject.toml uv.lock requirements.txt .gitignore .python-version .env.example
git commit -m "chore: configure reproducible NLP environment"
git add src scripts/download_data.py data/raw_manifest.json docs/DATASET_CARD.md docs/dataset_candidates.csv
git commit -m "feat: add pinned download of Spanish Fake News Corpus"
git add src/inf8239_u02/audit.py scripts/audit_data.py tests/test_data_contract.py
git commit -m "feat: audit duplicates and source leakage"
git add src/inf8239_u02/modeling.py src/inf8239_u02/evaluation.py scripts/train_text.py scripts/run_all.py tests
git commit -m "feat: compare baseline and two text pipelines"
git add docs/error_categories.csv reports notebooks app scripts/embeddings_network.py
git commit -m "docs: add executed notebook, reports and error analysis"
git add README.md docs/AI_USE.md
git commit -m "docs: README and AI use statement"
git tag u02-ejercicio03
```

## Uso de IA
Ver `docs/AI_USE.md`.

## Referencias
- Gómez-Adorno, H., Posadas-Durán, J. P., Bel Enguix, G., & Porto Capetillo, C. (2021). Overview of FakeDeS at IberLEF 2021: Fake news detection in Spanish shared task. *Procesamiento del Lenguaje Natural, 67*, 223–231.
- Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The elements of statistical learning* (2.ª ed.). Springer.
- Posadas-Durán, J. P., Gómez-Adorno, H., Sidorov, G., & Moreno Escobar, J. J. (2019). Detection of fake news in a new corpus for the Spanish language. *Journal of Intelligent & Fuzzy Systems, 36*(5), 4869–4876. https://doi.org/10.3233/JIFS-179034
