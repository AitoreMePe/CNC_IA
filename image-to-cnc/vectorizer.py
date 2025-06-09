import argparse
import vtracer
from pathlib import Path
import cv2
import numpy as np

# cv2 y numpy ya no son necesarios para la vectorización básica,
# pero los dejamos comentados para un futuro pre-procesamiento avanzado.
# import cv2
# import numpy as np

def preprocess_sketch(input_path: Path, output_dir: Path):
    """
    Pre-processes a sketch-like image to extract a clean, single-pixel-wide skeleton.
    Returns the path to the processed image.
    """
    print("Activado modo boceto. Aplicando pipeline final con esqueletización (v4)...")
    
    # Cargar la imagen en escala de grises
    img = cv2.imread(str(input_path), cv2.IMREAD_GRAYSCALE)
    
    # --- Pipeline de Pre-procesamiento v4 (Esqueletización) ---

    # 1. Suavizado (usamos la configuración agresiva v3 como base)
    img_denoised = cv2.bilateralFilter(img, d=9, sigmaColor=150, sigmaSpace=150)

    # 2. Umbral adaptativo para obtener una máscara binaria.
    img_bw = cv2.adaptiveThreshold(
        img_denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, blockSize=35, C=4
    )
    
    # 3. Invertir la imagen. La esqueletización espera objetos blancos sobre fondo negro.
    img_inverted = cv2.bitwise_not(img_bw)
    
    # 4. ESQUELETIZACIÓN: Reducir las formas a su línea central de 1 píxel.
    print("Adelgazando líneas a 1 píxel (esqueletización)...")
    skeleton = cv2.ximgproc.thinning(img_inverted)
    
    # 5. Invertir de nuevo para que vtracer reciba líneas negras sobre fondo blanco.
    skeleton_inverted = cv2.bitwise_not(skeleton)

    # Guardar la imagen final del esqueleto para poder revisarla
    processed_path = output_dir / f"{input_path.stem}_preprocessed_v4_skeleton.png"
    cv2.imwrite(str(processed_path), skeleton_inverted)
    print(f"Imagen de esqueleto (v4) guardada en: {processed_path}")
    
    return processed_path

def vectorize_image(input_path, output_path, is_sketch=False):
    """
    Toma una ruta de imagen de entrada y la convierte en un archivo SVG.
    """
    print(f"Iniciando vectorización para: {input_path}")
    
    input_file = Path(input_path)
    output_file = Path(output_path)

    if not input_file.is_file():
        print(f"Error: El archivo de entrada no se encuentra en '{input_file}'")
        return
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    vectorization_input = input_file

    if is_sketch:
        vectorization_input = preprocess_sketch(input_file, output_file.parent)

    print(f"Procesando '{vectorization_input.name}' con vtracer... (esto puede tardar un momento)")
    
    # Usamos la función correcta para la versión de la librería instalada.
    # Esta versión no parece aceptar parámetros de configuración complejos.
    vtracer.convert_image_to_svg_py(
        str(vectorization_input),
        str(output_file)
    )

    print(f"¡Vectorización completada! Archivo guardado en: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Transforma una imagen (PNG, JPG, etc.) a un archivo vectorial SVG para CNC.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("input_image", help="Ruta de la imagen a procesar.")
    parser.add_argument(
        "-o", "--output", 
        help="Ruta del archivo SVG de salida (por defecto: 'output/input_image_name.svg')."
    )
    parser.add_argument(
        "--sketch",
        action="store_true",
        help="Activa el modo de pre-procesamiento para bocetos y dibujos a lápiz."
    )
    
    args = parser.parse_args()

    # Crear un nombre de archivo de salida por defecto si no se proporciona
    if not args.output:
        input_p = Path(args.input_image)
        # El directorio de salida ahora es el mismo que el del script
        output_dir = Path("output")
        args.output = output_dir / f"{input_p.stem}.svg"

    vectorize_image(args.input_image, args.output, args.sketch) 