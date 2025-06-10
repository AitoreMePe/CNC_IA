# Image to CNC: Un Framework de IA para la Vectorización de Bocetos

Este repositorio documenta la creación de un sistema completo para convertir bocetos a lápiz en archivos vectoriales (SVG) listos para máquinas CNC o plotters. Lo que comenzó como un simple script se ha convertido en un completo framework de Inteligencia Artificial que incluye:

1.  **Una herramienta de trazado interactivo (`interactive_tracer.py`)**: Permite a un usuario trazar con precisión un boceto, generando datos de entrenamiento de alta calidad en el proceso.
2.  **Un sistema de preparación de datos (`data_preparator.py`)**: Unifica los datos generados por el usuario con datasets públicos para crear un conjunto de datos masivo.
3.  **Un modelo de Deep Learning (`train_model.py`)**: Una red neuronal (U-Net) que aprende a identificar los trazos en una imagen a partir de los datos.
4.  **Una herramienta de predicción (`predict.py`)**: Utiliza el modelo entrenado para convertir automáticamente nuevos bocetos en SVG.

---

## Estructura del Proyecto

```
/
|-- .gitignore                 <-- Archivo para ignorar ficheros no deseados en Git.
|-- datasets/
|   |-- 1_raw_downloads/       <-- Aquí se guardan los datasets públicos descargados.
|   |-- 3_user_generated/      <-- El trazador interactivo guarda aquí tus propios trazados.
|
|-- image-to-cnc/
|   |-- interactive_tracer.py  <-- Herramienta para trazar manualmente y generar datos.
|   |-- data_preparator.py     <-- Script para preparar los datos y crear el manifiesto.
|   |-- train_model.py         <-- Script para entrenar el modelo de IA.
|   |-- predict.py             <-- Script para usar la IA y predecir sobre nuevas imágenes.
|   |-- output/                <-- Carpeta donde se guardan las predicciones del modelo.
|
|-- manifest.json              <-- (Generado) Lista unificada de todos los datos de entrenamiento.
|-- best_model.pth             <-- (Generado) El modelo de IA una vez entrenado.
|-- README.md                  <-- Esta guía.
|-- requirements.txt           <-- Lista de dependencias de Python.
```

---

## Instalación

1.  **Clona el repositorio (si aún no lo has hecho):**
    ```bash
    git clone <URL_DEL_REPOSITORIO>
    cd <NOMBRE_DEL_REPOSITORIO>
    ```

2.  **Instala las dependencias de Python.** Es **crucial** instalar la versión de PyTorch que sea compatible con tu GPU (CUDA) para poder entrenar el modelo. El siguiente comando desinstalará cualquier versión existente e instalará la correcta para CUDA 11.8.

    Abre una terminal y ejecuta:
    ```bash
    pip uninstall torch torchvision torchaudio -y
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    pip install -r requirements.txt
    ```

    *Nota: Si tienes una versión de CUDA diferente, visita la [página oficial de PyTorch](https://pytorch.org/get-started/locally/) para encontrar el comando adecuado para tu sistema.*

---

## Flujo de Trabajo Completo: De la Imagen a la IA

### Fase 1: (Opcional) Generar tus propios datos

Si quieres entrenar a la IA con tus propios ejemplos, usa el trazador interactivo. Cada vez que lo uses, crearás una "lección" perfecta para el modelo.

1.  Ejecuta el script apuntando a una imagen de un boceto:
    ```bash
    python image-to-cnc/interactive_tracer.py "ruta/a/tu/imagen.png"
    ```
2.  En la ventana que aparece:
    - **Mantén pulsado el clic izquierdo** para "pintar" sobre una línea.
    - **Suelta el clic** para que el algoritmo encuentre el camino y lo trace.
3.  Cierra la ventana al terminar. Se guardará un SVG de salida y, lo más importante, una copia de los datos de entrenamiento en `datasets/3_user_generated/`.

### Fase 2: Preparar el Conjunto de Datos

Este paso es obligatorio antes del primer entrenamiento. El script `data_preparator.py` recorre todos los datasets (los tuyos y los públicos que hayas descargado en `datasets/1_raw_downloads/`) y crea un archivo `manifest.json`. Este manifiesto es un índice que le dice al script de entrenamiento dónde encontrar cada par de imagen de entrada y su correspondiente imagen de salida (el trazado limpio).

-   Para ejecutarlo, simplemente corre el siguiente comando en la terminal:
    ```bash
    python image-to-cnc/data_preparator.py
    ```
    Este proceso puede tardar un poco si tienes muchos datos.

### Fase 3: Entrenar el Modelo de IA

Aquí es donde ocurre la magia. El script `train_model.py` cargará los datos usando el `manifest.json` y entrenará la red neuronal U-Net. Utilizará tu GPU para acelerar el proceso masivamente.

-   Para iniciar el entrenamiento, ejecuta:
    ```bash
    python image-to-cnc/train_model.py
    ```
-   El script imprimirá el progreso (épocas y pérdida/loss). El objetivo es que la "pérdida" disminuya.
-   Cuando termine, el mejor modelo se guardará como `best_model.pth` en la raíz del proyecto.

### Fase 4: Usar la IA para Vectorizar una Imagen

Una vez que tienes un `best_model.pth`, puedes usar `predict.py` para realizar una vectorización automática en una nueva imagen.

1.  Asegúrate de que el archivo `best_model.pth` está en la raíz del proyecto.
2.  Ejecuta el script de predicción, pasándole la ruta de la nueva imagen que quieres procesar:
    ```bash
    python image-to-cnc/predict.py "ruta/a/tu/nueva_imagen.png"
    ```
3.  El script cargará el modelo, procesará la imagen y generará dos archivos en la carpeta `image-to-cnc/output/`:
    -   `predicted_mask.png`: La imagen de los trazos que la IA "cree" que debe haber.
    -   `final_vector.svg`: El archivo vectorial final, listo para tu CNC.

---

*Este proyecto fue desarrollado en colaboración. ¡Disfruta creando!* 