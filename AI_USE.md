# Declaración de uso de herramientas de IA · Ejercicio 03

## Herramienta
- **Claude (Anthropic)**, interfaz web claude.ai, dentro de un proyecto de la asignatura que contiene el programa de INF-8239 y el manual de laboratorios de la Unidad 01.
- Fecha de uso: octubre de 2026.

## Prompts relevantes
1. Se compartieron los proyectos base `INF8239_U02_NLP_Proyecto_Base_uv.zip` y `INF8239_U02_CV_Proyecto_Base_uv.zip` junto con el enunciado y la rúbrica de los Ejercicios 03 y 04, con la instrucción: *"Vamos a trabajar los archivos por separado"*.
2. Ante la pregunta de cómo iniciar, se indicó: *"Como digan las indicaciones o con lo que sea más conveniente si la indicación no la tiene"*.
3. *"Continuar"*, para completar notebook, documentación, declaración y entrega.

## Qué produjo la IA
- Propuesta de los dos corpus candidatos y verificación de sus licencias en las páginas oficiales.
- Código de descarga con commit fijado, auditoría, partición sin duplicados, pipelines, evaluación (bootstrap, McNemar, validación por grupos de medios, evaluación externa y ablación), figuras, pruebas, notebook y documentación.
- Borrador de las categorías de los 20 errores (`docs/error_categories.csv`).

## Partes verificadas por la estudiante
> Marque solo lo que haya comprobado personalmente.

- [ ] Ejecuté `uv sync`, `uv run pytest -q` y `uv run python scripts/run_all.py` en mi equipo y obtuve las mismas métricas que figuran en el README.
- [ ] Comprobé en GitHub que la licencia del corpus es CC BY 4.0 y que el commit fijado existe.
- [ ] Leí los 20 textos completos de `reports/error_analysis_20.csv` y confirmé o corregí cada categoría.
- [ ] Revisé que `source`, `link` e `id` no llegan al modelo (`LEAKAGE_COLUMNS` y la aserción en `train_text.py`).
- [ ] Verifiqué las referencias bibliográficas citadas.
- [ ] Puedo explicar cada archivo de `src/` y `scripts/`.

## Correcciones realizadas durante el desarrollo
| Problema detectado | Corrección |
|---|---|
| La auditoría mostró que el corpus v1.0 reemplaza cifras por `*NUMBER*` y el v2.0 conserva dígitos. | Se añadió `preprocess()` dentro del pipeline para unificar ambas formas y una ablación que mide su efecto en el corpus externo. |
| La primera regla de selección (máximo F1 en validación) elegía un modelo por una diferencia de 0.006, inferior a su variabilidad. | Se adoptó la regla de un error estándar (Hastie et al., 2009). **Transparencia:** el cambio se hizo después de una primera ejecución que ya mostraba resultados de prueba; la regla nueva elige el modelo con **menor** F1 en prueba, por lo que no favorece el resultado reportado. |
| Una prueba de `preprocess` fallaba por diferencias de espacios. | Se compararon tokens en lugar de cadenas. |
| `scripts/run_all.py` fallaba con `ModuleNotFoundError: scripts`. | Se agregó la raíz del proyecto a `sys.path`. |
| Leyendas y etiquetas superpuestas en figuras. | Se reubicaron leyendas y valores. |
| Avisos de Ruff (orden de importaciones, `noqa` innecesario). | Corregidos con `ruff check --fix` y revisión manual. |

## Responsabilidad
La estudiante conserva la responsabilidad sobre los datos, el código, las referencias y las conclusiones, y declara poder explicar cualquier fragmento entregado.
