import torch
import os
from PIL import Image
from torchvision import transforms
import argparse

# Importamos la arquitectura del modelo desde nuestro script de entrenamiento
from train_model import UNet, CONFIG

def predict(model, image_path, device, threshold):
    """
    Carga una imagen, la procesa y devuelve la predicción del modelo.
    """
    model.eval() # Poner el modelo en modo de evaluación
    
    # Cargar y transformar la imagen de entrada
    transform = transforms.Compose([
        transforms.Resize((CONFIG["image_size"], CONFIG["image_size"])),
        transforms.ToTensor(),
        transforms.Grayscale(), # Asegurarnos de que es de 1 canal
    ])
    
    image = Image.open(image_path).convert("L")
    input_tensor = transform(image).unsqueeze(0).to(device) # Añadir batch dimension

    with torch.no_grad():
        output = model(input_tensor)
        # Aplicamos una función sigmoide para convertir los logits a probabilidades (0-1)
        output_probs = torch.sigmoid(output)
        # Binarizamos con el umbral que nos pasen como parámetro
        output_binary = (output_probs > threshold).float()

    # Devolvemos tanto las probabilidades crudas como el resultado binarizado
    return output_probs.squeeze(0).cpu(), output_binary.squeeze(0).cpu()

def tensor_to_image(tensor):
    """
    Convierte un tensor de PyTorch a una imagen de Pillow.
    """
    # El tensor tiene 1 canal, lo quitamos para tener (H, W)
    tensor = tensor.squeeze(0) 
    # Multiplicamos por 255 y convertimos a un tipo de dato de imagen
    image_np = tensor.numpy() * 255
    image_pil = Image.fromarray(image_np.astype('uint8'), 'L')
    return image_pil

def main():
    parser = argparse.ArgumentParser(description="Usa el modelo U-Net entrenado para limpiar un boceto.")
    parser.add_argument('--input', type=str, required=True, help="Ruta a la imagen de boceto de entrada.")
    parser.add_argument('--output', type=str, required=True, help="Ruta para guardar la imagen limpia de salida. Se guardarán dos versiones: '_heatmap.png' y '_binary.png'.")
    parser.add_argument('--model_path', type=str, default=os.path.join(CONFIG["checkpoint_dir"], "best_model.pth"), help="Ruta al checkpoint del modelo entrenado.")
    parser.add_argument('--threshold', type=float, default=0.5, help="Umbral de confianza para binarizar la imagen de salida (0.0 a 1.0).")
    args = parser.parse_args()

    # Cargar el modelo entrenado
    device = CONFIG['device']
    print(f"Usando dispositivo: {device}")
    
    model = UNet(n_channels=1, n_classes=1).to(device)
    
    if not os.path.exists(args.model_path):
        print(f"Error: No se encuentra el archivo del modelo en '{args.model_path}'")
        print("Asegúrate de haber entrenado el modelo primero con 'train_model.py'.")
        return

    model.load_state_dict(torch.load(args.model_path, map_location=device))
    print(f"Modelo cargado desde '{args.model_path}'")

    # Realizar la predicción
    print(f"Procesando '{args.input}' con un umbral de {args.threshold}...")
    output_probs, output_tensor = predict(model, args.input, device, args.threshold)
    
    # Imprimir estadísticas de las probabilidades y guardarlas en un archivo
    stats_text = (
        f"Estadísticas de la salida (probabilidades) para '{args.input}':\n"
        f"  -> Mín: {output_probs.min():.4f}\n"
        f"  -> Máx: {output_probs.max():.4f}\n"
        f"  -> Media: {output_probs.mean():.4f}\n"
        f"Umbral aplicado: {args.threshold}\n"
    )
    print(stats_text)

    # Definir rutas de salida
    base_output, ext = os.path.splitext(args.output)
    heatmap_path = f"{base_output}_heatmap.png"
    binary_path = f"{base_output}_binary.png"
    stats_path = f"{base_output}_stats.txt"

    # Asegurarse de que el directorio de salida exista
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
    # Guardar las estadísticas en un archivo
    with open(stats_path, 'w') as f:
        f.write(stats_text)
    print(f"Estadísticas guardadas en '{stats_path}'")

    # Guardar la imagen de salida binaria
    output_image_binary = tensor_to_image(output_tensor)
    output_image_binary.save(binary_path)
    print(f"Imagen binarizada (umbral {args.threshold}) guardada en '{binary_path}'")
    
    # Guardar el mapa de calor de probabilidades
    output_image_heatmap = tensor_to_image(output_probs)
    output_image_heatmap.save(heatmap_path)
    print(f"Mapa de calor (confianza del modelo) guardado en '{heatmap_path}'")

if __name__ == "__main__":
    main() 