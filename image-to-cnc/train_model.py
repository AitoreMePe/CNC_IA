import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, ImageDraw
import numpy as np
from tqdm import tqdm
import svgwrite
from svgpathtools import svg2paths

# --- CÁLCULO DE RUTAS DINÁMICO ---
# Esto hace que el script sea robusto, sin importar desde dónde se ejecute.
# Localiza la ruta del script actual.
script_dir = os.path.dirname(os.path.abspath(__file__))
# Sube un nivel para llegar a la raíz del proyecto (la carpeta 'CNC').
project_root = os.path.dirname(script_dir)


# --- CONFIGURACIÓN ---
# Esta sección contendrá todos los hiperparámetros y configuraciones
# para que sea fácil experimentar y ajustar el modelo.
CONFIG = {
    "manifest_path": os.path.join(project_root, "manifest.json"),
    "base_dir": project_root, # El directorio base para las imágenes es la raíz
    "epochs": 50,
    "batch_size": 2, # Reducido para imágenes más grandes
    "learning_rate": 0.001,
    "image_size": 1024, # Aumentado para alta resolución
    "validation_split": 0.1, # 10% de los datos para validación
    "checkpoint_path": os.path.join(project_root, "best_model.pth"),
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    # Volvemos a activar los workers para aprovechar el Threadripper
    "num_workers": 8
}

# --- 1. CLASE DATASET ---
# Esta clase cargará los datos de nuestro manifiesto y los preparará
# para el entrenamiento. La parte más importante es la transformación
# de los SVG de salida en una imagen 'target' que el modelo pueda aprender.

class SketchDataset(Dataset):
    """Carga pares de imágenes de boceto y sus vectores limpios."""
    
    def __init__(self, manifest_entries, base_dir, image_size):
        self.entries = manifest_entries
        self.base_dir = base_dir
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            # Podríamos añadir más transformaciones (data augmentation) aquí
        ])

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        
        # Cargar la imagen de entrada (boceto sucio)
        input_img_path = os.path.join(self.base_dir, entry['input_image_path'])
        # Asegurarnos de que la imagen se carga en escala de grises para consistencia
        input_image = Image.open(input_img_path).convert('L') 
        input_tensor = self.transform(input_image)
        
        # Rasterizar el SVG de salida para crear la imagen 'target'
        output_vec_path = os.path.join(self.base_dir, entry['output_vector_path'])
        target_tensor = self.rasterize_svg(output_vec_path, (CONFIG["image_size"], CONFIG["image_size"]))

        return input_tensor, target_tensor

    def rasterize_svg(self, svg_path, size):
        # Esta es una función clave. Convierte un SVG en una imagen (tensor).
        # Usaremos una implementación simple por ahora.
        # NOTA: Esta función es una simplificación y podría necesitar mejoras.
        # Por ejemplo, usar una librería más robusta como 'cairosvg'.
        width, height = size
        img = Image.new('L', (width, height), 'white')
        drawing = svgwrite.Drawing(size=(width, height))
        
        try:
            # Leemos los trazos del JSON que creamos antes
            with open(svg_path, 'r') as f:
                strokes = json.load(f)

            # Dibujamos cada trazo en un objeto de imagen de Pillow
            draw = ImageDraw.Draw(img)
            for stroke in strokes:
                # Los puntos deben ser tuplas
                points_tuples = [tuple(p) for p in stroke]
                if len(points_tuples) > 1:
                    draw.line(points_tuples, fill='black', width=2) # Grosor aumentado

            # Convertimos la imagen de Pillow a un tensor de PyTorch
            return self.transform(img)

        except (FileNotFoundError, json.JSONDecodeError, TypeError):
            # Si hay un error o el SVG no existe, devolvemos una imagen en blanco
            return torch.zeros((1, width, height))


# --- 2. ARQUITECTURA DEL MODELO (U-Net) ---
# Implementación de una U-Net estándar.
# La red aprende a reducir la imagen a sus características esenciales (codificador)
# y luego reconstruye una imagen limpia a partir de ellas (decodificador).
# Las "skip connections" permiten que la información de alta resolución 
# fluya directamente del codificador al decodificador, preservando los detalles.

class DoubleConv(nn.Module):
    """(Convolution => [BatchNorm] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Ajustar el tamaño si es necesario
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2,
                                  diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=1, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


# --- 3. BUCLE DE ENTRENAMIENTO ---

def train_model(model, data_loader, criterion, optimizer, device):
    model.train() # Poner el modelo en modo de entrenamiento
    total_loss = 0
    
    for inputs, targets in tqdm(data_loader, desc="Entrenando"):
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Backward pass y optimización
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    return total_loss / len(data_loader)

def validate_model(model, data_loader, criterion, device):
    model.eval() # Poner el modelo en modo de evaluación
    total_loss = 0
    with torch.no_grad(): # No necesitamos calcular gradientes en validación
        for inputs, targets in tqdm(data_loader, desc="Validando"):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
    return total_loss / len(data_loader)


# --- 4. FUNCIÓN PRINCIPAL ---
# Se elimina la función run_training y se pone todo en el main guard

if __name__ == '__main__':
    print("--- Script de Entrenamiento Iniciado ---")
    
    # Comprobación explícita de CUDA
    if not torch.cuda.is_available():
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!! ADVERTENCIA: CUDA no está disponible. Usando CPU. !!!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

    device = CONFIG['device']
    print(f"Usando dispositivo: {device}")
    
    with open(CONFIG["manifest_path"], 'r') as f:
        manifest = json.load(f)
        
    # Eliminamos el filtro para usar TODOS los datos del manifiesto.
    # ¡Esto es clave para un buen entrenamiento!
    trainable_manifest = [
        m for m in manifest if m.get('output_vector_path')
    ]
        
    print(f"Encontradas {len(trainable_manifest)} muestras entrenables.")

    split_idx = int(len(trainable_manifest) * (1 - CONFIG["validation_split"]))
    train_manifest = trainable_manifest[:split_idx]
    val_manifest = trainable_manifest[split_idx:]
    print(f"Dividiendo en {len(train_manifest)} para entrenamiento y {len(val_manifest)} para validación.")

    train_dataset = SketchDataset(train_manifest, CONFIG['base_dir'], CONFIG['image_size'])
    val_dataset = SketchDataset(val_manifest, CONFIG['base_dir'], CONFIG['image_size'])
    
    # Se añade pin_memory=True para optimizar la transferencia a la GPU
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=CONFIG['num_workers'], pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=CONFIG['num_workers'], pin_memory=True)
    
    model = UNet().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
    
    best_val_loss = float('inf')

    print("\n--- ¡Comenzando Bucle de Entrenamiento! ---\n")
    for epoch in range(1, CONFIG["epochs"] + 1):
        print(f"\n--- Epoch {epoch}/{CONFIG['epochs']} ---")
        
        train_loss = train_model(model, train_loader, criterion, optimizer, device)
        val_loss = validate_model(model, val_loader, criterion, device)
        
        print(f"Epoch {epoch} -> Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Guardar el mejor modelo
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), CONFIG["checkpoint_path"])
            print(f"Nuevo mejor modelo guardado en {CONFIG['checkpoint_path']} (Val Loss: {val_loss:.4f})")
            
    print("\n--- ¡Entrenamiento completado! ---")
    print(f"El mejor modelo se ha guardado en: {CONFIG['checkpoint_path']}") 