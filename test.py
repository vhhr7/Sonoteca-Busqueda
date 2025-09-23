# -*- coding: utf-8 -*-
import os

# Evita warnings de paralelismo de tokenizers si existen modelos
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("NO_PROXY", "127.0.0.1,localhost")
os.environ.setdefault("no_proxy", "127.0.0.1,localhost")

import gradio as gr

# Verificación de versiones de dependencias
try:
    import gradio, gradio_client, sentence_transformers, faiss, numpy, scipy, sklearn, torch
    print("gradio:", gradio.__version__)
    print("gradio-client:", gradio_client.__version__)
    print("sentence-transformers:", sentence_transformers.__version__)
    print("faiss:", faiss.__version__)
    print("numpy:", numpy.__version__)
    print("scipy:", scipy.__version__)
    print("scikit-learn:", sklearn.__version__)
    print("torch:", torch.__version__)
except Exception as e:
    print("Error verificando versiones:", e)

# Función mínima de eco

def eco(txt: str) -> str:
    return txt

# Interfaz mínima
iface = gr.Interface(fn=eco, inputs="text", outputs="text", title="Smoke Test Gradio")

if __name__ == "__main__":
    # show_api=False evita construir el esquema OpenAPI (previene el error TypeError: 'bool' is not iterable)
    iface.launch(show_api=False, server_name="0.0.0.0", server_port=7860, share=False)
