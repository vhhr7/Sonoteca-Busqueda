import os

APP_ENV = os.getenv("APP_ENV", "development")

if APP_ENV == "production":
    INDEX_DIR = "/mnt/user/nextcloud/vicherrera/files/Sonoteca/Index"
else:
    INDEX_DIR = "/Volumes/Libreria/Sonoteca/Index"

os.makedirs(INDEX_DIR, exist_ok=True)

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
                registros.append((ruta_completa, nombre))

    # Guardar lista en archivo de texto (ruta \t nombre)
    with open(salida_txt, "w", encoding="utf-8") as out:
        for ruta, n in registros:
            out.write(f"{ruta}\t{n}\n")

    print(f"✅ Lista generada con {len(registros)} archivos.")
    print(f"📄 Guardado en: {salida_txt}")
    return registros


if __name__ == "__main__":
    if APP_ENV == "production":
        ruta = "/mnt/user/nextcloud/vicherrera/files/Sonoteca"
    else:
        ruta = "/Volumes/Libreria/Sonoteca"
    preparar_lista_sonoteca(ruta)