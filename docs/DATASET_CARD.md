# Dataset Card · The Spanish Fake News Corpus

## Identificación
- **Nombre:** The Spanish Fake News Corpus, versiones 1.0 y 2.0.
- **Fuente original:** https://github.com/jpposadas/FakeNewsCorpusSpanish
- **Commit utilizado:** `732f4870abd8d3f5ab4956fd92b040a7913614e3` (fijado en `src/inf8239_u02/config.py`).
- **Responsables:** J. P. Posadas-Durán (ESIME Zacatenco, IPN), H. Gómez-Adorno (IIMAS, UNAM), G. Sidorov (CIC, IPN) y colaboradores.
- **Versión o fecha:** v1.0 recopilada de enero a julio de 2018; v2.0 de noviembre de 2020 a marzo de 2021 (FakeDeS, IberLEF 2021).
- **Licencia:** Creative Commons Attribution 4.0 International (CC BY 4.0). Permite uso académico con atribución.
- **Idioma:** Español (predominio de español de México; la v2.0 incluye otras variantes regionales).
- **Integridad:** hashes SHA-256 de los archivos crudos y procesados en `data/raw_manifest.json`.

## Propósito y variable objetivo
Investigación sobre detección de noticias falsas basada en estilo. Variable objetivo `label`
(original `Category`): `fake` o `true`. En este proyecto se usa para responder si un clasificador
basado solo en texto distingue ambas clases y cuánto generaliza a medios y periodos no vistos.

## Diccionario de datos (esquema estandarizado en `data/processed/*.csv`)

| Columna | Tipo | Descripción | Valores o unidad | ¿Predictor? |
|---|---|---|---|---|
| id | texto | Identificador original dentro de cada archivo | Entero como texto | No (identificador) |
| split | texto | Origen del registro | train, development, external_v2 | No (control de partición) |
| label | texto | Clase objetivo, unificada desde `Category`/`CATEGORY` | fake, true | Objetivo |
| topic | texto | Tema asignado por los autores | 9 temas en v1.0, 7 en v2.0 | No (solo análisis) |
| source | texto | Medio que publicó la noticia | 150 medios en v1.0 | **No: fuga** |
| link | texto | URL original | URL | **No: fuga** |
| text | texto | Texto completo; incluye el titular en el 98 % de los casos | Español, mediana ≈ 315 palabras | **Sí (único)** |

La columna `Headline` original no se usa por separado para no duplicar el titular.

## Procedimiento de obtención
`uv run python scripts/download_data.py` descarga `train.xlsx`, `development.xlsx` y `test.xlsx`
desde `raw.githubusercontent.com` en el commit fijado, verifica los hashes contra el manifiesto
y escribe `data/processed/corpus.csv` (v1.0, 971 filas) y `data/processed/external_v2.csv`
(v2.0, 572 filas). No se usa ninguna ruta local ni de Google Drive.

## Calidad observada (`reports/audit_summary.json`)
- Sin textos vacíos ni etiquetas nulas. Clases balanceadas: 480 fake y 491 true.
- 5 duplicados normalizados: 4 entre train y development y 1 dentro de development; ninguno con etiqueta contradictoria. Se eliminan de la prueba.
- 147 de 150 medios publican una sola clase; un modelo con solo el medio obtiene F1 macro ≈ 0.93.
- La v1.0 sustituye las cifras por `*NUMBER*`; la v2.0 conserva los dígitos.
- 70 textos mencionan el nombre de su propio medio.

## Población cubierta y excluida
Cubre noticias en línea en español de 2018 (v1.0) y 2020-2021 (v2.0), con fuerte presencia de
medios mexicanos y sitios satíricos (p. ej., El Dizque, El Ruinaversal) dentro de la clase fake.
Excluye redes sociales en v1.0, otros periodos, otros dominios temáticos y noticias que mezclan
información verdadera y falsa (la etiqueta es binaria).

## Riesgos, sesgos y usos prohibidos
- **Sesgo de fuente:** la etiqueta está casi determinada por el medio; un modelo puede aprender a reconocer medios en lugar de veracidad.
- **Sátira como falsedad:** parte de la clase fake es sátira explícita, más fácil de detectar que la desinformación real.
- **Deriva temporal y regional:** desempeño menor en la v2.0 (COVID-19, otros países).
- **Prohibido:** usar el modelo para censurar, sancionar o etiquetar públicamente noticias o medios reales, o como sustituto de la verificación periodística.

## Cita
Posadas-Durán, J. P., Gómez-Adorno, H., Sidorov, G., & Moreno Escobar, J. J. (2019). Detection of fake news in a new corpus for the Spanish language. *Journal of Intelligent & Fuzzy Systems, 36*(5), 4869–4876. https://doi.org/10.3233/JIFS-179034
