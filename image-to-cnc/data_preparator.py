import os
import json
import glob
from tqdm import tqdm
import numpy as np
import xml.etree.ElementTree as ET
import re
import csv
from PIL import Image

# Definición de rutas principales
# Nos aseguramos de que las rutas se construyen desde la ubicación del script
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_DIR = os.path.join(BASE_DIR, "datasets", "1_raw_downloads")
USER_GEN_DIR = os.path.join(BASE_DIR, "datasets", "3_user_generated")
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "datasets", "2_processed_data")

def parse_svg(svg_path):
    """
    Una función de utilidad para extraer trazos de archivos SVG complejos.
    Maneja etiquetas <polyline>, <path> y <line>.
    Devuelve una lista de trazos, donde cada trazo es una lista de puntos (x, y).
    """
    try:
        tree = ET.parse(svg_path)
        root = tree.getroot()
        namespaces = {'svg': 'http://www.w3.org/2000/svg'}
        all_strokes = []

        # Expresión regular para encontrar todos los números en una cadena
        coord_regex = re.compile(r"[-+]?\d*\.\d+|[-+]?\d+")

        # 1. Buscar etiquetas <polyline>
        for polyline in root.findall('.//svg:polyline', namespaces):
            points_str = polyline.get('points', '').strip()
            if not points_str: continue
            points = [float(p) for p in points_str.replace(',', ' ').split()]
            stroke = []
            # Aseguramos que procesamos los puntos en pares, truncando si la lista es impar
            for i in range(0, len(points) // 2 * 2, 2):
                stroke.append([points[i], points[i+1]])
            if stroke:
                all_strokes.append(stroke)
        
        # 2. Buscar etiquetas <path>
        for path in root.findall('.//svg:path', namespaces):
            d_attr = path.get('d', '')
            if not d_attr: continue
            points = [float(p) for p in coord_regex.findall(d_attr)]
            stroke = []
            # Asumimos que los puntos vienen en pares (x,y) y truncamos si es impar
            for i in range(0, len(points) // 2 * 2, 2):
                stroke.append([points[i], points[i+1]])
            if stroke:
                all_strokes.append(stroke)

        # 3. Buscar etiquetas <line>
        for line in root.findall('.//svg:line', namespaces):
            try:
                x1 = line.get('x1')
                y1 = line.get('y1')
                x2 = line.get('x2')
                y2 = line.get('y2')
                if all(v is not None for v in [x1, y1, x2, y2]):
                    all_strokes.append([[float(x1), float(y1)], [float(x2), float(y2)]])
            except (ValueError, TypeError):
                continue

        return all_strokes
    except (ET.ParseError, FileNotFoundError):
        return []

def process_user_generated_data(output_path):
    """
    Procesa los datos que hemos generado con la herramienta interactiva.
    El formato es: carpetas con input.png y output.svg.
    """
    print("Procesando datos generados por el usuario...")
    manifest = []
    
    # Buscamos todas las carpetas de sesión
    session_folders = glob.glob(os.path.join(USER_GEN_DIR, "*"))
    
    for session_path in tqdm(session_folders, desc="  -> Sesiones de usuario"):
        input_img_path = os.path.join(session_path, "input.png")
        output_svg_path = os.path.join(session_path, "output.svg")
        
        if os.path.exists(input_img_path) and os.path.exists(output_svg_path):
            # Extraemos los trazos del SVG a un formato JSON universal
            vector_data = parse_svg(output_svg_path)
            
            if not vector_data:
                continue

            # Creamos un nombre de archivo único para nuestro JSON de vectores
            session_name = os.path.basename(session_path)
            json_output_filename = f"user_{session_name}.json"
            json_output_path = os.path.join(output_path, json_output_filename)
            
            # Guardamos los vectores en el nuevo archivo JSON
            with open(json_output_path, 'w') as f:
                json.dump(vector_data, f)
            
            # Añadimos la entrada al manifiesto
            manifest.append({
                "source_dataset": "user_generated",
                "input_image_path": os.path.relpath(input_img_path, BASE_DIR),
                "output_vector_path": os.path.relpath(json_output_path, BASE_DIR)
            })
            
    return manifest


def process_sketchbench_data(output_path):
    """
    Procesa el dataset SketchBench utilizando el archivo de mapeo sketch_tags.csv.
    """
    print("Procesando SketchBench (con la lógica final del CSV)...")
    manifest = []
    
    data_dir = os.path.join(RAW_DATA_DIR, "sketchbench", "Benchmark_Dataset")
    csv_path = os.path.join(data_dir, "sketch_tags.csv")
    rough_dir = os.path.join(data_dir, "Rough")
    gt_dir = os.path.join(data_dir, "GT")

    if not os.path.exists(csv_path):
        print(f"  -> Error: No se encontró el archivo 'sketch_tags.csv' en '{data_dir}'.")
        print("  -> Saltando SketchBench.")
        return manifest

    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader) 

            for row in tqdm(rows, desc="  -> Muestras de SketchBench"):
                if row.get('Cleaned', '').strip().lower() == 'yes':
                    base_name = row['Name']
                    
                    input_img_path = None
                    possible_png = os.path.join(rough_dir, "PNG", f"{base_name}.png")
                    possible_jpg = os.path.join(rough_dir, "JPG", f"{base_name}.jpg")
                    
                    if os.path.exists(possible_png):
                        input_img_path = possible_png
                    elif os.path.exists(possible_jpg):
                        input_img_path = possible_jpg
                    else:
                        continue
                        
                    gt_svg_pattern = os.path.join(gt_dir, f"{base_name}*.svg")
                    found_svgs = glob.glob(gt_svg_pattern)

                    # Si no se encuentra ningún SVG, podría ser un dato faltante. Lo saltamos.
                    if not found_svgs:
                        # Opcional: imprimir un aviso para saber qué se está saltando.
                        # print(f"  -> Aviso: No se encontró SVG para el base_name '{base_name}'. Saltando.")
                        continue
                    
                    # De los encontrados, priorizamos el que contenga '_cleaned.svg'
                    gt_svg_path = next((path for path in found_svgs if '_cleaned.svg' in os.path.basename(path)), None)

                    # Si después de buscar no encontramos una versión '_cleaned', saltamos esta entrada.
                    if not gt_svg_path:
                        continue

                    vector_data = parse_svg(gt_svg_path)

                    if not vector_data:
                        continue

                    json_output_filename = f"sketchbench_{base_name}.json"
                    json_output_path = os.path.join(output_path, json_output_filename)

                    with open(json_output_path, 'w') as f:
                        json.dump(vector_data, f)
                    
                    manifest.append({
                        "source_dataset": "sketchbench",
                        "input_image_path": os.path.relpath(input_img_path, BASE_DIR),
                        "output_vector_path": os.path.relpath(json_output_path, BASE_DIR)
                    })
    except Exception as e:
        print(f"  -> Ocurrió un error procesando el CSV: {e}")

    return manifest

def process_sketchy_data(output_path):
    """
    Procesa el dataset Sketchy.
    Busca pares de bocetos SVG y sus correspondientes fotos renderizadas.
    """
    print("Procesando Sketchy Database...")
    manifest = []
    
    base_sketchy_dir = os.path.join(RAW_DATA_DIR, "sketchy")
    sketches_base_dir = os.path.join(base_sketchy_dir, "sketches-06-04", "sketches")
    rendered_photos_dir = os.path.join(base_sketchy_dir, "rendered_256x256", "256x256", "photo", "tx_000000000000")

    if not os.path.isdir(sketches_base_dir) or not os.path.isdir(rendered_photos_dir):
        print("  -> No se encontraron las carpetas necesarias para Sketchy. Saltando...")
        return manifest

    # Obtenemos la lista de todas las categorías (subcarpetas)
    try:
        categories = [d for d in os.listdir(sketches_base_dir) if os.path.isdir(os.path.join(sketches_base_dir, d))]
    except FileNotFoundError:
        print(f"  -> Error: No se pudo encontrar el directorio de bocetos: {sketches_base_dir}")
        return manifest

    for category in tqdm(categories, desc="  -> Categorías de Sketchy"):
        category_path = os.path.join(sketches_base_dir, category)
        svg_files = glob.glob(os.path.join(category_path, "*.svg"))

        for svg_path in svg_files:
            base_name = os.path.splitext(os.path.basename(svg_path))[0]
            
            # La foto renderizada debería tener el mismo nombre base, pero con extensión .jpg
            # y estar en la carpeta de fotos renderizadas.
            photo_path = os.path.join(rendered_photos_dir, category, f"{base_name}.jpg")

            if os.path.exists(photo_path):
                # En este caso, el SVG es la *entrada* y la foto es la *salida*
                # Aunque no es un vector, lo guardamos en 'output_vector_path' 
                # para mantener la consistencia del manifiesto.
                manifest.append({
                    "source_dataset": "sketchy",
                    "category": category,
                    "input_image_path": os.path.relpath(svg_path, BASE_DIR),
                    "output_vector_path": os.path.relpath(photo_path, BASE_DIR)
                })

    return manifest

def process_quickdraw_data(output_path):
    """
    Procesa el dataset Quick, Draw!.
    Convierte los arrays de numpy en imágenes PNG.
    """
    print("Procesando Quick, Draw!...")
    manifest = []
    quickdraw_dir = os.path.join(RAW_DATA_DIR, "quickdraw")
    
    if not os.path.isdir(quickdraw_dir):
        print("  -> No se encontró la carpeta 'quickdraw'. Saltando...")
        return manifest
        
    npy_files = glob.glob(os.path.join(quickdraw_dir, "*.npy"))
    
    # Crear una subcarpeta para las imágenes de Quick, Draw!
    quickdraw_output_dir = os.path.join(output_path, "quickdraw_images")
    os.makedirs(quickdraw_output_dir, exist_ok=True)

    for npy_file in tqdm(npy_files, desc="  -> Archivos de Quick, Draw!"):
        try:
            category_name = os.path.splitext(os.path.basename(npy_file))[0].replace("full_numpy_bitmap_", "")
            data = np.load(npy_file)
            
            # Cada fila en el archivo .npy es un dibujo
            for i, img_array in enumerate(data):
                # Los datos son un array plano de 784, lo convertimos a 28x28
                img_matrix = img_array.reshape(28, 28)
                
                # Convertimos el array de numpy a una imagen de Pillow y la guardamos
                img = Image.fromarray(img_matrix.astype(np.uint8), 'L')
                img_output_path = os.path.join(quickdraw_output_dir, f"{category_name}_{i}.png")
                img.save(img_output_path)

                manifest.append({
                    "source_dataset": "quickdraw",
                    "category": category_name,
                    "input_image_path": os.path.relpath(img_output_path, BASE_DIR),
                    "output_vector_path": None # No hay Ground Truth para Quick, Draw!
                })

        except Exception as e:
            print(f"  -> Error procesando el archivo {os.path.basename(npy_file)}: {e}")
            continue
            
    return manifest


def main():
    """
    Función principal que orquesta todo el proceso de preparación.
    """
    print("--- Iniciando la preparación del dataset ---")
    
    # Creamos el directorio de salida si no existe.
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    
    # Procesamos cada dataset, esta vez activando todos.
    user_manifest = process_user_generated_data(PROCESSED_DATA_DIR)
    sketchbench_manifest = process_sketchbench_data(PROCESSED_DATA_DIR)
    sketchy_manifest = process_sketchy_data(PROCESSED_DATA_DIR) 
    quickdraw_manifest = process_quickdraw_data(PROCESSED_DATA_DIR)

    # Combinamos todos los manifiestos en uno solo
    full_manifest = user_manifest + sketchbench_manifest + sketchy_manifest + quickdraw_manifest
    
    # Guardamos el manifiesto final en la raíz del proyecto
    manifest_path = os.path.join(BASE_DIR, "manifest.json")
    
    with open(manifest_path, 'w') as f:
        json.dump(full_manifest, f, indent=4)
        
    print(f"\n--- Proceso completado ---")
    print(f"Manifiesto final creado en: {manifest_path}")
    print(f"Total de muestras procesadas: {len(full_manifest)}")


if __name__ == "__main__":
    main() 