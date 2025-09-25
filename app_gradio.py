# -*- coding: utf-8 -*-
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # silencia warning de tokenizers
from pydub import AudioSegment
import tempfile

# Selección de rutas por entorno
APP_ENV = os.getenv("APP_ENV", "development")
if APP_ENV == "production":
    BASE_INDEX_DIR = "/sonoteca/Index"
else:
    BASE_INDEX_DIR = "/Volumes/Libreria/Sonoteca/Index"

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import gradio as gr

# ==== Config ====
LISTA_TXT = os.path.join(BASE_INDEX_DIR, "lista_sonoteca.txt")          # generado por preparar_lista.py (formato: RUTA \t NOMBRE)
INDEX_FAISS = os.path.join(BASE_INDEX_DIR, "sonoteca.index")            # índice persistente (si existe, se carga)
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
TOP_K_DEFAULT = 10

# ==== Estado global (se cargan una vez) ====
MODEL = None
INDEX = None
RUTAS = []
NOMBRES = []

def cargar_lista(lista_txt):
    rutas, nombres = [], []
    with open(lista_txt, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) >= 2:
                ruta, nombre = parts[0], parts[1]
            else:
                # fallback: sin tab, usar basename como nombre
                ruta = parts[0]
                nombre = os.path.splitext(os.path.basename(parts[0]))[0].replace("_", " ").replace("-", " ")
            if os.path.exists(ruta):
                rutas.append(ruta)
                nombres.append(nombre)
    return rutas, nombres

def cargar_modelo():
    global MODEL
    if MODEL is None:
        MODEL = SentenceTransformer(MODEL_NAME)
    return MODEL

def crear_o_cargar_indice(nombres):
    """Carga FAISS si el tamaño coincide; si no, re-embebe y reindexa."""
    global INDEX
    model = cargar_modelo()

    if os.path.exists(INDEX_FAISS):
        try:
            idx = faiss.read_index(INDEX_FAISS)
            if idx.ntotal == len(nombres):
                INDEX = idx
                return INDEX
        except Exception:
            pass  # si falla, recalculamos

    # Recalcular
    embs = model.encode(nombres, convert_to_numpy=True, normalize_embeddings=True)
    dim = embs.shape[1]
    idx = faiss.IndexFlatIP(dim)  # producto interno (equiv. coseno al normalizar)
    idx.add(embs)
    os.makedirs(os.path.dirname(INDEX_FAISS), exist_ok=True)
    faiss.write_index(idx, INDEX_FAISS)
    INDEX = idx
    return INDEX

def inicializar():
    """Carga lista, modelo e índice una sola vez."""
    global RUTAS, NOMBRES
    if not os.path.exists(LISTA_TXT):
        raise FileNotFoundError(f"No existe {LISTA_TXT}. Ejecuta primero preparar_lista.py con APP_ENV={APP_ENV}")
    RUTAS, NOMBRES = cargar_lista(LISTA_TXT)
    cargar_modelo()
    crear_o_cargar_indice(NOMBRES)

# ===== Lógica de búsqueda =====
def buscar(prompt, k):
    if not prompt.strip():
        return [], None, gr.update(choices=[], value=None)

    q_emb = MODEL.encode([prompt], convert_to_numpy=True, normalize_embeddings=True)
    D, I = INDEX.search(q_emb, int(k))
    filas = []
    opciones = []
    for rank, idx in enumerate(I[0]):
        # Puede devolver -1 si no hay suficientes resultados; filtrar
        if idx < 0 or idx >= len(NOMBRES):
            continue
        ruta = RUTAS[idx]
        nombre = NOMBRES[idx]
        score = float(D[0][rank])
        filas.append([rank + 1, nombre, ruta, round(score, 3)])
        opciones.append(f"{rank+1}. {nombre}")
    # Seleccionar por defecto el top-1 si hay
    selected = opciones[0] if opciones else None
    # Para el reproductor, archivo top-1 si existe
    audio_path = filas[0][2] if filas else None
    return filas, audio_path, gr.update(choices=opciones, value=selected)

def elegir_y_reproducir(eleccion, tabla):
    """eleccion: '1. Nombre', tabla: puede llegar como list[list] o como pandas.DataFrame
    Devuelve la ruta del elemento elegido o None.
    """
    # Si no hay elección, no hacemos nada
    if not eleccion:
        return None

    # Normalizar la tabla a una lista de filas [[rank, nombre, ruta, score], ...]
    filas = None
    try:
        # Si es un DataFrame de pandas (como lo entrega gr.Dataframe en 3.x)
        import pandas as pd  # import local para no obligar en tiempo de import
        if isinstance(tabla, pd.DataFrame):
            if tabla.empty:
                return None
            filas = tabla.values.tolist()
    except Exception:
        pass

    # Si ya viene como lista de listas
    if filas is None:
        if isinstance(tabla, list):
            # Puede venir como list de dicts o list de lists
            if not tabla:
                return None
            if isinstance(tabla[0], dict):
                # Ordenar columnas en el orden esperado
                cols = ["#", "Nombre", "Ruta", "Score"]
                filas = [[row.get(c) for c in cols] for row in tabla]
            else:
                filas = tabla
        else:
            # Tipo no soportado
            return None

    # Parsear el índice seleccionado
    try:
        idx = int(str(eleccion).split(".")[0]) - 1
    except Exception:
        return None

    if idx < 0 or idx >= len(filas):
        return None

    # La columna 3 es la ruta según nuestro formato [rank, nombre, ruta, score]
    ruta = filas[idx][2]
    # Si es AIFF, convertir a WAV temporal
    if ruta and ruta.lower().endswith(".aiff"):
        audio = AudioSegment.from_file(ruta, format="aiff")
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        audio.export(tmp.name, format="wav")
        return tmp.name
    return ruta

# ===== UI Gradio =====
with gr.Blocks(title="Buscador Sonoteca (títulos)") as demo:
    gr.Markdown("# 🔎 Buscador de Sonidos por Texto (solo títulos)\nBusca por lenguaje natural, ve la ruta y escucha el archivo.")

    with gr.Row():
        prompt = gr.Textbox(label="Escribe tu búsqueda", placeholder="ej. viento en bosque oscuro, ramas")
        topk = gr.Slider(1, 50, value=TOP_K_DEFAULT, step=1, label="Resultados (Top-K)")

    buscar_btn = gr.Button("Buscar", variant="primary")

    with gr.Row():
        resultados = gr.Dataframe(
            headers=["#","Nombre","Ruta","Score"],
            datatype=["number","str","str","number"],
            row_count=(0, "dynamic"),
            wrap=True,
            label="Resultados",
            interactive=False
        )

    with gr.Row():
        opcion = gr.Dropdown(choices=[], label="Elegir para reproducir", interactive=True)
        reproducir_btn = gr.Button("▶️ Reproducir seleccionado")

    audio_out = gr.Audio(label="Reproductor", autoplay=True)

    # Eventos
    def do_search(q, k):
        filas, audio_path, dd = buscar(q, k)
        # si hay audio top-1, lo cargamos en el player
        return filas, audio_path, dd

    buscar_btn.click(
        do_search,
        inputs=[prompt, topk],
        outputs=[resultados, audio_out, opcion],
        preprocess=True
    )

    reproducir_btn.click(
        elegir_y_reproducir,
        inputs=[opcion, resultados],
        outputs=[audio_out]
    )

# Lanzar
if __name__ == "__main__":
    inicializar()
    # share=False para local; si quieres link público, usa share=True
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)

# -*- coding: utf-8 -*-
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # silencia warning de tokenizers

# Selección de rutas por entorno
APP_ENV = os.getenv("APP_ENV", "development")
if APP_ENV == "production":
    BASE_INDEX_DIR = "/sonoteca/Index"
else:
    BASE_INDEX_DIR = "/Volumes/Libreria/Sonoteca/Index"

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import gradio as gr

# ==== Config ====
LISTA_TXT = os.path.join(BASE_INDEX_DIR, "lista_sonoteca.txt")          # generado por preparar_lista.py (formato: RUTA \t NOMBRE)
INDEX_FAISS = os.path.join(BASE_INDEX_DIR, "sonoteca.index")            # índice persistente (si existe, se carga)
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
TOP_K_DEFAULT = 10

# ==== Estado global (se cargan una vez) ====
MODEL = None
INDEX = None
RUTAS = []
NOMBRES = []

def cargar_lista(lista_txt):
    rutas, nombres = [], []
    with open(lista_txt, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) >= 2:
                ruta, nombre = parts[0], parts[1]
            else:
                # fallback: sin tab, usar basename como nombre
                ruta = parts[0]
                nombre = os.path.splitext(os.path.basename(parts[0]))[0].replace("_", " ").replace("-", " ")
            if os.path.exists(ruta):
                rutas.append(ruta)
                nombres.append(nombre)
    return rutas, nombres

def cargar_modelo():
    global MODEL
    if MODEL is None:
        MODEL = SentenceTransformer(MODEL_NAME)
    return MODEL

def crear_o_cargar_indice(nombres):
    """Carga FAISS si el tamaño coincide; si no, re-embebe y reindexa."""
    global INDEX
    model = cargar_modelo()

    if os.path.exists(INDEX_FAISS):
        try:
            idx = faiss.read_index(INDEX_FAISS)
            if idx.ntotal == len(nombres):
                INDEX = idx
                return INDEX
        except Exception:
            pass  # si falla, recalculamos

    # Recalcular
    embs = model.encode(nombres, convert_to_numpy=True, normalize_embeddings=True)
    dim = embs.shape[1]
    idx = faiss.IndexFlatIP(dim)  # producto interno (equiv. coseno al normalizar)
    idx.add(embs)
    os.makedirs(os.path.dirname(INDEX_FAISS), exist_ok=True)
    faiss.write_index(idx, INDEX_FAISS)
    INDEX = idx
    return INDEX

def inicializar():
    """Carga lista, modelo e índice una sola vez."""
    global RUTAS, NOMBRES
    if not os.path.exists(LISTA_TXT):
        raise FileNotFoundError(f"No existe {LISTA_TXT}. Ejecuta primero preparar_lista.py con APP_ENV={APP_ENV}")
    RUTAS, NOMBRES = cargar_lista(LISTA_TXT)
    cargar_modelo()
    crear_o_cargar_indice(NOMBRES)

# ===== Lógica de búsqueda =====
def buscar(prompt, k):
    if not prompt.strip():
        return [], None, gr.update(choices=[], value=None)

    q_emb = MODEL.encode([prompt], convert_to_numpy=True, normalize_embeddings=True)
    D, I = INDEX.search(q_emb, int(k))
    filas = []
    opciones = []
    for rank, idx in enumerate(I[0]):
        # Puede devolver -1 si no hay suficientes resultados; filtrar
        if idx < 0 or idx >= len(NOMBRES):
            continue
        ruta = RUTAS[idx]
        nombre = NOMBRES[idx]
        score = float(D[0][rank])
        filas.append([rank + 1, nombre, ruta, round(score, 3)])
        opciones.append(f"{rank+1}. {nombre}")
    # Seleccionar por defecto el top-1 si hay
    selected = opciones[0] if opciones else None
    # Para el reproductor, archivo top-1 si existe
    audio_path = filas[0][2] if filas else None
    return filas, audio_path, gr.update(choices=opciones, value=selected)

def elegir_y_reproducir(eleccion, tabla):
    """eleccion: '1. Nombre', tabla: puede llegar como list[list] o como pandas.DataFrame
    Devuelve la ruta del elemento elegido o None.
    """
    # Si no hay elección, no hacemos nada
    if not eleccion:
        return None

    # Normalizar la tabla a una lista de filas [[rank, nombre, ruta, score], ...]
    filas = None
    try:
        # Si es un DataFrame de pandas (como lo entrega gr.Dataframe en 3.x)
        import pandas as pd  # import local para no obligar en tiempo de import
        if isinstance(tabla, pd.DataFrame):
            if tabla.empty:
                return None
            filas = tabla.values.tolist()
    except Exception:
        pass

    # Si ya viene como lista de listas
    if filas is None:
        if isinstance(tabla, list):
            # Puede venir como list de dicts o list de lists
            if not tabla:
                return None
            if isinstance(tabla[0], dict):
                # Ordenar columnas en el orden esperado
                cols = ["#", "Nombre", "Ruta", "Score"]
                filas = [[row.get(c) for c in cols] for row in tabla]
            else:
                filas = tabla
        else:
            # Tipo no soportado
            return None

    # Parsear el índice seleccionado
    try:
        idx = int(str(eleccion).split(".")[0]) - 1
    except Exception:
        return None

    if idx < 0 or idx >= len(filas):
        return None

    # La columna 3 es la ruta según nuestro formato [rank, nombre, ruta, score]
    ruta = filas[idx][2]
    # Si es AIFF, convertir a WAV temporal
    if ruta and ruta.lower().endswith(".aiff"):
        audio = AudioSegment.from_file(ruta, format="aiff")
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        audio.export(tmp.name, format="wav")
        return tmp.name
    return ruta

# ===== UI Gradio =====
with gr.Blocks(title="Buscador Sonoteca (títulos)") as demo:
    gr.Markdown("# 🔎 Buscador de Sonidos por Texto (solo títulos)\nBusca por lenguaje natural, ve la ruta y escucha el archivo.")

    with gr.Row():
        prompt = gr.Textbox(label="Escribe tu búsqueda", placeholder="ej. viento en bosque oscuro, ramas")
        topk = gr.Slider(1, 50, value=TOP_K_DEFAULT, step=1, label="Resultados (Top-K)")

    buscar_btn = gr.Button("Buscar", variant="primary")

    with gr.Row():
        resultados = gr.Dataframe(
            headers=["#","Nombre","Ruta","Score"],
            datatype=["number","str","str","number"],
            row_count=(0, "dynamic"),
            wrap=True,
            label="Resultados",
            interactive=False
        )

    with gr.Row():
        opcion = gr.Dropdown(choices=[], label="Elegir para reproducir", interactive=True)
        reproducir_btn = gr.Button("▶️ Reproducir seleccionado")

    audio_out = gr.Audio(label="Reproductor", autoplay=True)

    # Eventos
    def do_search(q, k):
        filas, audio_path, dd = buscar(q, k)
        # si hay audio top-1, lo cargamos en el player
        return filas, audio_path, dd

    buscar_btn.click(
        do_search,
        inputs=[prompt, topk],
        outputs=[resultados, audio_out, opcion],
        preprocess=True
    )

    reproducir_btn.click(
        elegir_y_reproducir,
        inputs=[opcion, resultados],
        outputs=[audio_out]
    )

# Lanzar
if __name__ == "__main__":
    inicializar()
    # share=False para local; si quieres link público, usa share=True
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
