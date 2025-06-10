import torch
import os
from PIL import Image
from torchvision import transforms
import argparse

# Importamos la arquitectura del modelo desde nuestro script de entrenamiento
# Asegúrate de que train_model.py está en el mismo directorio o en el python path
from train_model import UNet, CONFIG as TRAIN_CONFIG

def predict(model, image_path, device, threshold, image_size):
    """
    Carga una imagen, la procesa y devuelve la predicción del modelo.
    """
    model.eval()
    
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Grayscale(),
    ])
    
    image = Image.open(image_path).convert("L")
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        output_probs = torch.sigmoid(output)
        output_binary = (output_probs > threshold).float()

    return output_probs.squeeze(0).cpu(), output_binary.squeeze(0).cpu()

def tensor_to_image(tensor):
    """
    Convierte un tensor de PyTorch a una imagen de Pillow.
    """
    tensor = tensor.squeeze(0)
    image_np = tensor.numpy() * 255
    image_pil = Image.fromarray(image_np.astype('uint8'), 'L')
    return image_pil

def main():
    parser = argparse.ArgumentParser(description="Usa el modelo U-Net de alta resolución para limpiar un boceto.")
    parser.add_argument('--input', type=str, required=True, help="Ruta a la imagen de boceto de entrada.")
    parser.add_argument('--output', type=str, required=True, help="Ruta base para guardar las imágenes de salida.")
    parser.add_argument('--model_path', type=str, default=os.path.join(TRAIN_CONFIG["checkpoint_dir"], "best_model.pth"), help="Ruta al checkpoint del modelo entrenado.")
    parser.add_argument('--threshold', type=float, default=0.5, help="Umbral de confianza para binarizar la imagen (0.0 a 1.0).")
    
    args = parser.parse_args()

    device = TRAIN_CONFIG['device']
    image_size = TRAIN_CONFIG['image_size']
    print(f"Usando dispositivo: {device} y tamaño de imagen: {image_size}x{image_size}")
    
    model = UNet(n_channels=1, n_classes=1).to(device)
    
    if not os.path.exists(args.model_path):
        print(f"Error: No se encuentra el archivo del modelo en '{args.model_path}'")
        return

    # Cargar el modelo asegurándose de que se mapea a la ubicación correcta (cpu o cuda)
    model.load_state_dict(torch.load(args.model_path, map_location=torch.device(device)))
    print(f"Modelo cargado desde '{args.model_path}'")

    print(f"Procesando '{args.input}' con un umbral de {args.threshold}...")
    output_probs, output_tensor = predict(model, args.input, device, args.threshold, image_size)
    
    stats_text = (
        f"Estadísticas (probabilidades) para '{args.input}':\n"
        f"  -> Mín: {output_probs.min():.4f}, Máx: {output_probs.max():.4f}, Media: {output_probs.mean():.4f}\n"
        f"Umbral aplicado: {args.threshold}\n"
    )
    print(stats_text)

    base_output, ext = os.path.splitext(args.output)
    heatmap_path = f"{base_output}_heatmap.png"
    binary_path = f"{base_output}_binary.png"
    stats_path = f"{base_output}_stats.txt"

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
    with open(stats_path, 'w') as f:
        f.write(stats_text)
    print(f"Estadísticas guardadas en '{stats_path}'")

    output_image_binary = tensor_to_image(output_tensor)
    output_image_binary.save(binary_path)
    print(f"Imagen binarizada (umbral {args.threshold}) guardada en '{binary_path}'")
    
    output_image_heatmap = tensor_to_image(output_probs)
    output_image_heatmap.save(heatmap_path)
    print(f"Mapa de calor guardado en '{heatmap_path}'")

if __name__ == "__main__":
    main() 