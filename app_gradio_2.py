# -*- coding: utf-8 -*-
# UI Gradio para búsqueda semántica por títulos y reproducción local

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # silencia warning de tokenizers
os.environ.setdefault("NO_PROXY", "127.0.0.1,localhost")
os.environ.setdefault("no_proxy", "127.0.0.1,localhost")

import gradio as gr
import gradio.routes as gr_routes
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# ====== Config ======
LISTA_TXT = "/Users/victorherrera/Documents/Scripts/Busqueda-Natural-Sonoteca-2/Index/lista_sonoteca.txt"  # ruta absoluta a la lista
INDEX_FAISS = "/Users/victorherrera/Documents/Scripts/Busqueda-Natural-Sonoteca-2/Index/sonoteca.index"    # índice FAISS persistente
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
TOP_K_DEFAULT = 10

# Si conoces la carpeta raíz de tus audios, puedes fijarla aquí.
# Si la dejas en None, la inferimos a partir de las rutas en lista_sonoteca.txt
ALLOWED_AUDIO_ROOT = None   # ej: "/Volumes/nextcloud/vicherrera/files/Sonoteca"

# ====== Estado global ======
MODEL = None
INDEX = None
RUTAS = []
NOMBRES = []

# ---------- Utilidades ----------
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
                ruta = parts[0]
                base = os.path.splitext(os.path.basename(ruta))[0]
                nombre = base.replace("_", " ").replace("-", " ")
            if os.path.exists(ruta):
                rutas.append(ruta)
                nombres.append(nombre)
    return rutas, nombres

def inferir_roots(rutas):
    """Obtiene raíces únicas para allowed_paths (máximo 4 para no sobrecargar)."""
    # Tomamos las 1–3 carpetas más altas (primeros 2 o 3 segmentos) como raíces aproximadas
    roots = set()
    for r in rutas[:5000]:  # sample para rendimiento
        try:
            partes = r.split(os.sep)
            if len(partes) > 3:
                root = os.sep.join(partes[:4])  # p.ej. /Volumes/nextcloud/vicherrera
            elif len(partes) > 2:
                root = os.sep.join(partes[:3])
            else:
                root = os.path.dirname(r)
            if root:
                roots.add(root)
        except Exception:
            pass
        if len(roots) >= 4:
            break
    return sorted(roots)

def cargar_modelo():
    global MODEL
    if MODEL is None:
        MODEL = SentenceTransformer(MODEL_NAME)
    return MODEL

def crear_o_cargar_indice(nombres):
    """Carga FAISS si coincide tamaño; si no, recalcula embeddings y reindexa."""
    global INDEX
    model = cargar_modelo()
    if os.path.exists(INDEX_FAISS):
        try:
            idx = faiss.read_index(INDEX_FAISS)
            if idx.ntotal == len(nombres):
                INDEX = idx
                print(f"📦 Índice cargado de {INDEX_FAISS} (items={idx.ntotal})")
                return INDEX
            else:
                print(f"ℹ️ Rehaciendo índice: idx={idx.ntotal}, nombres={len(nombres)}")
        except Exception as e:
            print(f"⚠️ No se pudo cargar índice existente: {e}. Se recalculará.")

    print("🎶 Generando embeddings de los títulos...")
    embs = model.encode(nombres, convert_to_numpy=True, normalize_embeddings=True)
    dim = embs.shape[1]
    idx = faiss.IndexFlatIP(dim)  # IP ≈ coseno al normalizar
    idx.add(embs)
    faiss.write_index(idx, INDEX_FAISS)
    print(f"📦 Índice guardado en {INDEX_FAISS}")
    INDEX = idx
    return INDEX

def inicializar():
    """Carga lista, modelo e índice una vez al inicio."""
    global RUTAS, NOMBRES
    if not os.path.exists(LISTA_TXT):
        raise FileNotFoundError(f"No existe {LISTA_TXT}. Ejecuta primero preparar_lista.py")
    RUTAS, NOMBRES = cargar_lista(LISTA_TXT)
    print(f"✅ Cargados {len(NOMBRES)} items")
    cargar_modelo()
    crear_o_cargar_indice(NOMBRES)

# ---------- Lógica de búsqueda ----------
def buscar_backend(prompt, k):
    """Devuelve filas para la tabla, ruta top-1 para player, y opciones del dropdown."""
    if not prompt or not prompt.strip():
        return [], None, gr.update(choices=[], value=None)

    q_emb = MODEL.encode([prompt], convert_to_numpy=True, normalize_embeddings=True)
    D, I = INDEX.search(q_emb, int(k))

    filas, opciones = [], []
    for rank, idx in enumerate(I[0]):
        if idx < 0 or idx >= len(NOMBRES):
            continue
        ruta = RUTAS[idx]
        nombre = NOMBRES[idx]
        score = float(D[0][rank])
        filas.append([rank + 1, nombre, ruta, round(score, 3)])
        opciones.append(f"{rank+1}. {nombre}")

    selected = opciones[0] if opciones else None
    audio_path = filas[0][2] if filas else None
    return filas, audio_path, gr.update(choices=opciones, value=selected)

def elegir_y_reproducir(eleccion, tabla):
    """
    eleccion: '1. Nombre'
    tabla: puede venir como list[list] o como pandas.DataFrame
    Devuelve la ruta (columna 3) o None.
    """
    if not eleccion:
        return None
    try:
        idx = int(eleccion.split(".")[0]) - 1
    except Exception:
        return None

    # Normalizar la tabla a lista de filas
    rows = None
    if hasattr(tabla, "empty") and hasattr(tabla, "values"):  # DataFrame
        if tabla.empty:
            return None
        rows = tabla.values.tolist()
    else:
        if not isinstance(tabla, list) or len(tabla) == 0:
            return None
        rows = tabla

    if idx < 0 or idx >= len(rows):
        return None

    try:
        ruta = rows[idx][2]  # [#, nombre, ruta, score]
    except Exception:
        return None
    return ruta

# ---------- UI Gradio ----------
with gr.Blocks(title="Buscador Sonoteca (títulos)") as demo:
    gr.Markdown("# 🔎 Buscador de Sonidos por Texto (solo títulos)\nBusca con lenguaje natural, ve la ruta y escucha el archivo.")

    with gr.Row():
        prompt = gr.Textbox(label="Escribe tu búsqueda", placeholder="ej. viento en bosque oscuro, ramas")
        topk = gr.Slider(1, 50, value=TOP_K_DEFAULT, step=1, label="Resultados (Top-K)")

    buscar_btn = gr.Button("Buscar", variant="primary")

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

    # Callbacks
    def do_search(q, k):
        filas, audio_path, dd = buscar_backend(q, k)
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

# Evita el bug del schema en algunas versiones
demo.get_api_info = lambda: {}
gr_routes.api_info = lambda *args, **kwargs: {}

# ---------- Lanzar servidor ----------
if __name__ == "__main__":
    inicializar()

    # Determinar allowed_paths para servir los audios directamente
    allowed = []
    if ALLOWED_AUDIO_ROOT and os.path.exists(ALLOWED_AUDIO_ROOT):
        allowed = [ALLOWED_AUDIO_ROOT]
    else:
        roots = inferir_roots(RUTAS)
        allowed = [r for r in roots if os.path.exists(r)]
        if not allowed:
            # como último recurso, la carpeta de trabajo
            allowed = [os.getcwd()]

    demo.launch(
        server_name="127.0.0.1",  # acceso local seguro
        server_port=7860,
        share=False,              # solo local; pon True si lo necesitas público
        show_api=False,           # evita el bug del schema
        allowed_paths=allowed
    )