from pathlib import Path

import joblib
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models/text_model.joblib"

st.title("INF-8239 · Detector académico de noticias falsas en español")
st.caption(
    "Demostración académica entrenada con The Spanish Fake News Corpus (2018). "
    "La predicción NO verifica hechos ni constituye una decisión automática."
)

if not MODEL_PATH.exists():
    st.error("No existe el modelo. Ejecute: uv run python scripts/train_text.py")
    st.stop()

model = joblib.load(MODEL_PATH)
text = st.text_area("Texto completo de la noticia (titular y cuerpo)", height=220)
if st.button("Clasificar"):
    if not text.strip():
        st.warning("Ingrese un texto.")
    else:
        prediction = model.predict([text])[0]
        st.metric("Clase predicha", str(prediction))
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba([text])[0]
            st.write({label: round(float(p), 3) for label, p in zip(model.classes_, proba)})
        st.info(
            "El modelo reconoce patrones de estilo y vocabulario del corpus de 2018; "
            "su desempeño baja con medios no vistos y con noticias más recientes."
        )
