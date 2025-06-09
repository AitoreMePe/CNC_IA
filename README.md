# Image to Vector Converter for CNC

Este proyecto contiene un conjunto de herramientas de Python para convertir imágenes rasterizadas (como PNG, JPG) en archivos vectoriales SVG, optimizados para su uso en máquinas CNC, cortadoras láser o plotters.

El proyecto ha evolucionado desde un script de conversión totalmente automático hasta una sofisticada herramienta de trazado interactivo asistida por IA.

---

## Herramientas

Hay dos herramientas principales en este proyecto:

1.  `vectorizer.py`: (Automático) Un script que intenta convertir una imagen de forma totalmente automática. Es ideal para logos o imágenes con bordes muy definidos. Para arte complejo como bocetos, el resultado puede variar.
2.  `interactive_tracer.py`: (Recomendado) Una herramienta interactiva que te permite guiar al algoritmo, combinando la inteligencia humana con la precisión de la máquina. Es la solución definitiva para obtener resultados de alta calidad a partir de cualquier tipo de imagen.

---

## Instalación

1.  **Clona o descarga el proyecto.**
2.  **Instala las dependencias.** Abre una terminal en la carpeta del proyecto y ejecuta:
    ```bash
    pip install opencv-contrib-python-headless matplotlib scikit-image svgwrite vtracer
    ```

---

## Uso de `interactive_tracer.py` (Recomendado)

Esta es la herramienta más potente y precisa. Te permite trazar líneas sobre la imagen y un algoritmo encontrará el camino de píxeles perfecto.

### Cómo Funciona

1.  **Ejecuta el script** desde tu terminal, pasándole la ruta de tu imagen:
    ```bash
    python image-to-cnc/interactive_tracer.py "ruta/a/tu/imagen.png"
    ```
2.  **Se abrirá una ventana** con tu imagen.
3.  **Dibuja los trazos:**
    - Mantén **pulsado el botón izquierdo** del ratón y "pinta" sobre una línea que quieras trazar. Verás una guía roja.
    - **Suelta el botón** para finalizar ese trazo. El script lo registrará.
    - Repite el proceso para todas las líneas que quieras en tu diseño final.
4.  **Cierra la ventana** cuando hayas terminado.
5.  **Revisa los resultados:**
    - El archivo final se guardará como `drag_output.svg` (o el nombre que especifiques con `-o`). Este es el archivo que usarás en tu CNC.
    - **¡Importante!** También se creará una carpeta `dataset/` con una subcarpeta para tu sesión. Dentro, encontrarás una copia de tu imagen (`input.png`) y tu trazado (`output.svg`). Cada vez que usas la herramienta, estás creando un ejemplo de entrenamiento perfecto para una futura IA que pueda automatizar este trabajo.

### Opciones

-   `-o` o `--output`: Especifica un nombre de archivo de salida diferente para tu SVG.
    ```bash
    python image-to-cnc/interactive_tracer.py mi_dibujo.png -o mi_dibujo_final.svg
    ```

---

## Uso de `vectorizer.py` (Automático)

Esta herramienta es útil para conversiones rápidas de imágenes simples.

### Cómo Funciona

Ejecuta el script desde la terminal, indicando la imagen de entrada.

```bash
python image-to-cnc/vectorizer.py "ruta/a/tu/imagen.png"
```

El resultado se guardará en la carpeta `output/`.

### Opciones

-   `--sketch`: Activa un modo de pre-procesamiento avanzado para bocetos y dibujos a lápiz. Intenta limpiar la imagen antes de vectorizarla.
    ```bash
    python image-to-cnc/vectorizer.py mi_boceto.png --sketch
    ```
-   `-o` o `--output`: Especifica un nombre de archivo de salida diferente.

---
*Este proyecto fue desarrollado en colaboración. ¡Disfruta creando!* 