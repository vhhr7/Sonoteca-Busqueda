from mutagen import File as MutagenFile
import os

APP_ENV = os.getenv("APP_ENV", "development")

if APP_ENV == "production":
    INDEX_DIR = "/sonoteca/Index"
else:
    INDEX_DIR = "/Volumes/Libreria/Sonoteca/Index"

os.makedirs(INDEX_DIR, exist_ok=True)

def extraer_comentario(ruta_completa: str) -> str:
    """
    Intenta extraer el 'comentario/description' de metadatos del archivo de audio,
    soportando MP3/ID3, FLAC/OGG Vorbis, WAV/BWF/AIFF (RIFF/AIFF chunks) según soporte de mutagen.
    Devuelve cadena vacía si no encuentra nada.
    """
    try:
        audio = MutagenFile(ruta_completa, easy=True)
        if not audio:
            return ""
        # Campos comunes (easy tags)
        for key in ("comment", "description", "COMM", "DESCRIPTION"):
            val = audio.tags.get(key) if hasattr(audio, "tags") and audio.tags else None
            if val:
                # mutagen easy tags suelen devolver lista
                if isinstance(val, (list, tuple)):
                    return str(val[0]).strip()
                return str(val).strip()
        # Para MP3 (ID3 avanzado) intenta COMM frames
        try:
            audio_full = MutagenFile(ruta_completa, easy=False)
            if hasattr(audio_full, "tags") and audio_full.tags:
                comms = [f.text[0] for f in audio_full.tags.values() if f.FrameID == "COMM" and getattr(f, "text", None)]
                if comms:
                    return str(comms[0]).strip()
        except Exception:
            pass
    except Exception:
        pass
    return ""

def preparar_lista_sonoteca(ruta_sonoteca, salida_txt=os.path.join(INDEX_DIR, "lista_sonoteca.txt")):
    """
    Recorre la carpeta de la sonoteca y genera una lista con ruta completa y nombre limpio.
    """
    registros = []
    extensiones_validas = (".wav", ".mp3", ".aiff", ".flac", ".ogg")

    for root, _, files in os.walk(ruta_sonoteca):
        for f in files:
            if f.lower().endswith(extensiones_validas):
                ruta_completa = os.path.join(root, f)
                nombre = os.path.splitext(f)[0]
                nombre = nombre.replace("_", " ").replace("-", " ")
                comentario = extraer_comentario(ruta_completa)
                registros.append((ruta_completa, nombre, comentario))

    # Guardar lista en archivo de texto (ruta \t nombre \t comentario)
    with open(salida_txt, "w", encoding="utf-8") as out:
        for ruta, n, c in registros:
            # Escapar tabs nuevos si existieran
            c = (c or "").replace("\t", " ").replace("\n", " ").strip()
            out.write(f"{ruta}\t{n}\t{c}\n")

    print(f"✅ Lista generada con {len(registros)} archivos.")
    print(f"📄 Guardado en: {salida_txt}")
    return registros


if __name__ == "__main__":
    if APP_ENV == "production":
        ruta = "/sonoteca"
    else:
        ruta = "/Volumes/Libreria/Sonoteca"
    preparar_lista_sonoteca(ruta)