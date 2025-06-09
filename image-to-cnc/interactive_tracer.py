import matplotlib.pyplot as plt
import numpy as np
import cv2
from pathlib import Path
import argparse
from skimage.graph import route_through_array
import svgwrite

# --- Almacenamiento Global ---
# Guardaremos las rutas que el usuario defina aquí
all_paths = []
# Guardaremos los puntos del camino actual que se está dibujando
current_path_points = []

# --- Clase para gestionar el Trazado Interactivo ---
class InteractiveTracer:
    def __init__(self, ax, cost_matrix):
        self.ax = ax
        self.cost_matrix = cost_matrix
        self.drawing = False
        self.current_user_path = []
        
        # Conectar los eventos del ratón
        self.cid_press = ax.figure.canvas.mpl_connect('button_press_event', self.on_press)
        self.cid_release = ax.figure.canvas.mpl_connect('button_release_event', self.on_release)
        self.cid_motion = ax.figure.canvas.mpl_connect('motion_notify_event', self.on_motion)

    def on_press(self, event):
        if not event.inaxes or event.button != 1: return
        self.drawing = True
        self.current_user_path = [(int(event.xdata), int(event.ydata))]
        # Dibuja un punto rojo para indicar el inicio
        self.ax.plot(event.xdata, event.ydata, 'ro', markersize=5)
        self.ax.figure.canvas.draw()

    def on_motion(self, event):
        if not event.inaxes or not self.drawing: return
        self.current_user_path.append((int(event.xdata), int(event.ydata)))
        # Dibuja la línea guía del usuario en tiempo real
        path_np = np.array(self.current_user_path)
        self.ax.plot(path_np[:, 0], path_np[:, 1], 'r-', linewidth=1, alpha=0.7)
        self.ax.figure.canvas.draw()
        
    def on_release(self, event):
        if not self.drawing or event.button != 1: return
        self.drawing = False
        if len(self.current_user_path) > 1:
            print(f"Trazo guía finalizado con {len(self.current_user_path)} puntos.")
            all_paths.append(list(self.current_user_path)) # Guardamos la ruta
        self.current_user_path = []

# --- Función de Coste ---
def create_cost_matrix(image_gray):
    """
    Crea una matriz donde los píxeles oscuros tienen un coste bajo y los claros un coste alto.
    """
    # Los valores más altos son más 'caros' de atravesar. Invertimos la imagen.
    # Añadimos un valor pequeño para evitar costes de cero.
    return (255 - image_gray) + 1

# --- Función de Exportación a SVG ---
def export_to_svg(final_routes, output_filename, width, height):
    """
    Guarda las rutas calculadas en un archivo SVG.
    """
    size = (f"{width}px", f"{height}px")
    dwg = svgwrite.Drawing(output_filename, profile='tiny', size=size)
    
    for route in final_routes:
        epsilon = 2.0 * cv2.arcLength(route, True) / 100
        approx_route = cv2.approxPolyDP(route, epsilon, False)
        
        if len(approx_route) > 1:
            # CORRECCIÓN FINAL Y DEFINITIVA:
            # Convertir cada coordenada a un entero nativo de Python.
            points = [(int(p[0][0]), int(p[0][1])) for p in approx_route]
            dwg.add(dwg.polyline(points, stroke="black", fill="none", stroke_width=1))
        
    dwg.save()
    print(f"¡SVG guardado como {output_filename}!")

def main(image_path, output_path):
    """
    Función principal que carga la imagen y configura la ventana interactiva.
    """
    img_path = Path(image_path)
    if not img_path.is_file():
        print(f"Error: No se encuentra la imagen en {img_path}")
        return

    # Cargar la imagen original
    img = cv2.imread(str(img_path))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Matplotlib usa RGB
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    height, width, _ = img.shape
    
    # 1. Crear matriz de coste
    cost_matrix = create_cost_matrix(img_gray)

    # 2. Configurar y mostrar la ventana interactiva
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(img_rgb)
    ax.set_title("Trazador por Arrastre")
    fig.text(0.5, 0.01, "Mantén pulsado Clic Izq. para dibujar. Suelta para finalizar. Cierra la ventana para exportar.", ha='center')

    # Instanciar nuestra clase de trazado
    tracer = InteractiveTracer(ax, cost_matrix)

    print("Ventana interactiva abierta.")
    print("1. Mantén PULSADO el clic izquierdo y dibuja sobre un trazo.")
    print("2. SUELTA el botón para finalizar ese trazo.")
    print("3. Repite para todas las líneas que desees.")
    print("4. Cuando termines, CIERRA LA VENTANA para procesar y guardar el SVG.")
    
    # Mostrar la ventana. El script se pausará aquí hasta que se cierre.
    plt.show()

    # 3. Procesar las rutas después de cerrar la ventana
    if all_paths:
        print(f"\nProcesando {len(all_paths)} caminos capturados...")
        final_routes = []
        for user_path in all_paths:
            # Simplificamos la entrada del usuario para tener puntos clave
            user_path_np = np.array(user_path, dtype=np.int32).reshape(-1, 1, 2)
            epsilon = 0.01 * cv2.arcLength(user_path_np, True)
            key_points = cv2.approxPolyDP(user_path_np, epsilon, False)

            full_precise_route = []
            for i in range(len(key_points) - 1):
                start_coords = key_points[i][0]
                end_coords = key_points[i+1][0]
                start_point = (start_coords[1], start_coords[0]) # (fila, col)
                end_point = (end_coords[1], end_coords[0])
                
                try:
                    route, _ = route_through_array(cost_matrix, start_point, end_point, fully_connected=True)
                    route_xy = np.array(route)[:, ::-1].reshape(-1, 1, 2).astype(np.int32)
                    if not full_precise_route:
                        full_precise_route.extend(route_xy)
                    else:
                        full_precise_route.extend(route_xy[1:])
                except Exception as e:
                    print(f"No se pudo encontrar una ruta: {e}")
            
            if full_precise_route:
                final_routes.append(np.array(full_precise_route))

        # 4. Exportar a SVG
        if final_routes:
            export_to_svg(final_routes, output_path, width, height)
        else:
            print("No se pudo generar ninguna ruta válida.")
    else:
        print("\nNo se ha capturado ningún camino. Saliendo.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Trazador vectorial interactivo por arrastre.")
    parser.add_argument("input_image", help="Ruta de la imagen a trazar.")
    parser.add_argument("-o", "--output", help="Ruta del archivo SVG de salida.", default="drag_output.svg")
    args = parser.parse_args()
    main(args.input_image, args.output) 