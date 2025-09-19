# -*- coding: utf-8 -*-
# Buscador semántico sobre títulos (rutas + reproducción con stop)
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # silencia el warning

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import subprocess
import shutil
import sys

LISTA_TXT = "lista_sonoteca.txt"
INDEX_FAISS = "sonoteca.index"
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

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
                nombre = os.path.splitext(os.path.basename(parts[0]))[0].replace("_", " ").replace("-", " ")
            if os.path.exists(ruta):
                rutas.append(ruta)
                nombres.append(nombre)
    return rutas, nombres

def cargar_o_crear_indice(nombres, model):
    # ¿Existe índice y coincide el tamaño? Cárgalo
    if os.path.exists(INDEX_FAISS):
        try:
            index = faiss.read_index(INDEX_FAISS)
            if index.ntotal == len(nombres):
                print(f"📦 Índice cargado de {INDEX_FAISS} (items={index.ntotal})")
                return index
            else:
                print(f"ℹ️ Rehaciendo índice: tamaño actual={index.ntotal}, títulos={len(nombres)}")
        except Exception as e:
            print(f"⚠️ No se pudo cargar índice existente: {e}. Se recalculará.")
    # Recalcular
    print("🎶 Generando embeddings de los títulos...")
    embs = model.encode(nombres, convert_to_numpy=True, normalize_embeddings=True)
    dim = embs.shape[1]
    index = faiss.IndexFlatIP(dim)  # coseno (porque normalizamos)
    index.add(embs)
    faiss.write_index(index, INDEX_FAISS)
    print(f"📦 Índice guardado en {INDEX_FAISS}")
    return index

def buscar(prompt, k, model, index, rutas, nombres):
    q_emb = model.encode([prompt], convert_to_numpy=True, normalize_embeddings=True)
    D, I = index.search(q_emb, k)
    out = []
    for rank, idx in enumerate(I[0]):
        out.append((rutas[idx], nombres[idx], float(D[0][rank])))
    return out

# -------- Reproducción --------
def tiene_afplay():
    return shutil.which("afplay") is not None

def tiene_ffplay():
    return shutil.which("ffplay") is not None

def reproducir_async(ruta):
    """
    Lanza la reproducción sin bloquear. Devuelve el proceso.
    macOS: afplay
    Alternativa: ffplay (requiere ffmpeg instalado)
    """
    if tiene_afplay():
        return subprocess.Popen(["afplay", ruta])
    elif tiene_ffplay():
        # -nodisp: sin ventana / -autoexit: sale al terminar
        return subprocess.Popen(["ffplay", "-nodisp", "-autoexit", ruta], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        print("⚠️ No encontré 'afplay' ni 'ffplay'. Instala ffmpeg o usa macOS con afplay.")
        return None

def parar_reproduccion(proc):
    if proc is None:
        return
    try:
        proc.terminate()
    except Exception:
        pass

if __name__ == "__main__":
    # Cargar lista
    if not os.path.exists(LISTA_TXT):
        print(f"❌ No existe {LISTA_TXT}. Primero ejecuta preparar_lista.py")
        sys.exit(1)

    rutas, nombres = cargar_lista(LISTA_TXT)
    print(f"✅ Cargados {len(nombres)} items desde {LISTA_TXT}")

    # Modelo
    print("🔄 Cargando modelo de embeddings...")
    model = SentenceTransformer(MODEL_NAME)

    # Índice
    index = cargar_o_crear_indice(nombres, model)

    # Loop interactivo
    proc_actual = None
    try:
        while True:
            q = input("\n🔍 Escribe tu búsqueda (o 'salir'): ").strip()
            if q.lower() == "salir":
                break

            resultados = buscar(q, k=10, model=model, index=index, rutas=rutas, nombres=nombres)
            if not resultados:
                print("🫥 Sin resultados.")
                continue

            print("\n🎧 Resultados:")
            for i, (ruta, nombre, score) in enumerate(resultados, 1):
                print(f"{i}. {nombre} (score={score:.3f})")
                print(f"   📂 {ruta}")

            while True:
                opcion = input("\n▶️ Acción: número=Reproducir | s=Stop | n=Nueva búsqueda | salir=Salir: ").strip().lower()
                if opcion == "salir":
                    raise KeyboardInterrupt  # salimos del programa
                if opcion == "n":
                    break
                if opcion == "s":
                    parar_reproduccion(proc_actual)
                    proc_actual = None
                    print("⏹️ Reproducción detenida.")
                    continue
                if opcion.isdigit():
                    idx = int(opcion) - 1
                    if 0 <= idx < len(resultados):
                        # detener lo que esté sonando
                        parar_reproduccion(proc_actual)
                        ruta_sel, nombre_sel, _ = resultados[idx]
                        print(f"🎵 Reproduciendo: {ruta_sel}")
                        proc_actual = reproducir_async(ruta_sel)
                    else:
                        print("⚠️ Número fuera de rango.")
                else:
                    print("ℹ️ Comando no reconocido. Usa número, 's', 'n' o 'salir'.")
    except KeyboardInterrupt:
        print("\n👋 Saliendo...")
    finally:
        parar_reproduccion(proc_actual)