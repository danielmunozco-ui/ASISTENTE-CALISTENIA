# Asistente de Calistenia (RAG con Neon + Gemini)

Chat en vivo que responde **solo con los documentos indexados** y cita la fuente.

**Cómo funciona**
1. La pregunta se convierte en un vector (embedding, Gemini).
2. Un agente decide si buscar y qué buscar en Neon (Postgres + pgvector).
3. Responde únicamente con los fragmentos encontrados y cita el documento.

**Archivos**
- `app.py`: la aplicación (Streamlit).
- `requirements.txt`: dependencias.

**Secrets** (se configuran en Streamlit Cloud → Advanced settings → Secrets; nunca en este repositorio):

```toml
NEON_URL = "postgresql://..."
GEMINI_API_KEY = "..."
# Opcionales
# TABLA = "frag_calistenia"
# LIMITE_DIARIO = 150
# APP_PASSWORD = "una-clave"
# CHAT_MODEL = "gemini-3.5-flash-lite"
```

La tabla `frag_calistenia` se crea e indexa con el cuaderno de Colab (`RAG_Calistenia_Neon_Agente.ipynb`).
