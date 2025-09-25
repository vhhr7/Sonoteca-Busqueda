import os
import subprocess, json, shlex

APP_ENV = os.getenv("APP_ENV", "development")

if APP_ENV == "production":
    INDEX_DIR = "/sonoteca/Index"
else:
    INDEX_DIR = "/Volumes/Libreria/Sonoteca/Index"

os.makedirs(INDEX_DIR, exist_ok=True)

def _get_first_tag(tags: dict, keys):
    if not tags:
        return ""
    for k in keys:
        if k in tags and tags[k]:
            val = tags[k]
            if isinstance(val, list):
                return str(val[0]).strip()
            return str(val).strip()
    return ""

def ffprobe_comment(path: str) -> str:
    """
    Extrae comentario/description usando ffprobe (ffmpeg) para soportar AIFF/BWF/WAV/MP3/FLAC/OGG de forma amplia.
    Devuelve cadena vacía si no hay metadatos o ffprobe falla.
    """
    try:
        # -v error: solo errores
        # -print_format json: salida JSON
        # -show_format -show_streams: trae tags del contenedor y de cada stream
        cmd = f"ffprobe -v error -print_format json -show_format -show_streams {shlex.quote(path)}"
        out = subprocess.check_output(cmd, shell=True, text=True)
        data = json.loads(out)

        # Tags a revisar (orden de preferencia). Incluye variantes comunes:
        # comment/description (ID3/Vorbis/FLAC), ICMT/ICOM/ANNO (RIFF/AIFF), bext:description (BWF)
        preferred_keys = [
            "comment", "description",
            "COMMENT", "DESCRIPTION",
            "ICMT", "ICOM", "ANNO", "NOTE",
            "bext_description", "bext:description",  # a veces mapeado así
        ]

        # 1) Tags a nivel de formato
        fmt_tags = (data.get("format") or {}).get("tags") or {}
        texto = _get_first_tag(fmt_tags, preferred_keys)
        if texto:
            return texto

        # 2) Tags en los streams (algunos contenedores los ponen allí)
        for s in data.get("streams") or []:
            s_tags = s.get("tags") or {}
            texto = _get_first_tag(s_tags, preferred_keys)
            if texto:
                return texto
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
                comentario = ffprobe_comment(ruta_completa)
                registros.append((ruta_completa, nombre, comentario))

    # Guardar lista en archivo de texto (ruta \t nombre \t comentario)
    with open(salida_txt, "w", encoding="utf-8") as out:
        for ruta, n, c in registros:
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