import os
import pandas as pd
import shutil
import random
from pathlib import Path
import cv2
import torch
import multiprocessing
from ultralytics import YOLO

# ----------------- CONFIGURAZIONE -----------------
# Percorsi
DATASET_DIR = 'Dataset'  # Cartella con le sottocartelle Mulo e NonMulo
CSV_FILE = os.path.join(DATASET_DIR, 'dataset.csv')
OUTPUT_DIR = 'YoloDataset'  # Cartella di output per il dataset in formato YOLO

# Parametri di addestramento
IMG_SIZE = 640
EPOCHS = 100
BATCH_SIZE = 16
PATIENCE = 20  # Early stopping se il modello non migliora dopo questo numero di epoche

# Funzione per ottenere le dimensioni dell'immagine
def get_image_dimensions(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return 640, 640  # Valori predefiniti se l'immagine non può essere letta
    return img.shape[1], img.shape[0]  # width, height

# Funzione per preparare un'immagine per YOLO
def prepare_image_for_yolo(img_path, is_mulo, subset="train"):
    full_img_path = os.path.join(DATASET_DIR, img_path)
    img_name = os.path.basename(img_path)
    
    # Copia l'immagine nella directory corretta
    dest_path = os.path.join(OUTPUT_DIR, 'images', subset, img_name)
    shutil.copy(full_img_path, dest_path)
    
    # Se è un mulo, crea il file di annotazione
    if is_mulo == 1:
        # Diverse varianti del percorso per trovare una corrispondenza
        img_path_variants = [
            img_path,                                      # Mulo/mulo1.png
            img_path.replace('\\', '/'),                   # Mulo/mulo1.png (se originariamente con \)
            'Dataset/' + img_path,                         # Dataset/Mulo/mulo1.png
            'Dataset/' + img_path.replace('\\', '/'),      # Dataset/Mulo/mulo1.png (se originariamente con \)
            os.path.basename(img_path)                     # mulo1.png
        ]
        
        # Cerca nel DataFrame tutte le varianti del percorso
        found = False
        for path_variant in img_path_variants:
            img_row = annotations_df[annotations_df['ImagePath'] == path_variant]
            if not img_row.empty and 'X' in img_row.columns and 'Y' in img_row.columns:
                found = True
                # Ottieni le dimensioni dell'immagine
                img_width, img_height = get_image_dimensions(full_img_path)
                
                # Ottieni X e Y dalle annotazioni
                x_center = img_row['X'].values[0] / img_width
                y_center = img_row['Y'].values[0] / img_height
                
                # Per YOLO, width e height sono stimati (usa valori ragionevoli)
                width = 0.25  # 25% della larghezza dell'immagine
                height = 0.25  # 25% dell'altezza dell'immagine
                
                # Limita i valori a [0, 1]
                x_center = max(0, min(1, x_center))
                y_center = max(0, min(1, y_center))
                width = max(0, min(1, width))
                height = max(0, min(1, height))
                
                # Scrivi nel formato YOLO: class x_center y_center width height
                label_path = os.path.join(OUTPUT_DIR, 'labels', subset, os.path.splitext(img_name)[0] + '.txt')
                with open(label_path, 'w') as f:
                    f.write(f'0 {x_center} {y_center} {width} {height}\n')
                break
        
        if not found:
            print(f"ATTENZIONE: Coordinate non trovate per {img_path}")
            # Se non troviamo coordinate, usiamo valori approssimativi al centro dell'immagine
            label_path = os.path.join(OUTPUT_DIR, 'labels', subset, os.path.splitext(img_name)[0] + '.txt')
            with open(label_path, 'w') as f:
                f.write(f'0 0.5 0.5 0.25 0.25\n')  # Valori centrali di default

def main():
    global annotations_df  # Rendiamo annotations_df globale per accedervi dalla funzione prepare_image_for_yolo
    
    # ----------------- PREPARAZIONE DATASET -----------------
    print("Preparazione del dataset...")

    # Crea le cartelle necessarie
    os.makedirs(os.path.join(OUTPUT_DIR, 'images', 'train'), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, 'images', 'val'), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, 'labels', 'train'), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, 'labels', 'val'), exist_ok=True)

    # Leggi il file CSV con le annotazioni
    if os.path.exists(CSV_FILE):
        annotations_df = pd.read_csv(CSV_FILE)
        print(f"File CSV caricato con {len(annotations_df)} annotazioni")
        
        # Stampa le prime righe del CSV per debug
        print("Prime righe del CSV:")
        print(annotations_df.head())
        print("Colonne del CSV:", annotations_df.columns.tolist())
        
        # Standardizza i percorsi nel DataFrame
        if 'ImagePath' in annotations_df.columns:
            # Converti backslash in forward slash per consistenza
            annotations_df['ImagePath'] = annotations_df['ImagePath'].str.replace('\\', '/')
            # Rimuovi 'Dataset/' dal percorso se presente
            annotations_df['ImagePath'] = annotations_df['ImagePath'].str.replace('Dataset/', '')
    else:
        print(f"ATTENZIONE: File CSV {CSV_FILE} non trovato, creando un dataset senza annotazioni di coordinate")
        annotations_df = pd.DataFrame(columns=['ImagePath', 'IsMulo', 'X', 'Y'])

    # Raccogli tutte le immagini
    all_images = []

    # Raccolta immagini dalla cartella Mulo
    mulo_dir = os.path.join(DATASET_DIR, 'Mulo')
    if os.path.exists(mulo_dir):
        mulo_images = [os.path.join('Mulo', img) for img in os.listdir(mulo_dir) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
        print(f"Trovate {len(mulo_images)} immagini nella cartella Mulo")
        all_images.extend([(img, 1) for img in mulo_images])  # 1 = è un mulo
        
        # Debug: controlla se alcune immagini sono nel CSV
        for i, img in enumerate(mulo_images[:5]):  # Solo le prime 5 per debug
            img_path_standard = img.replace('\\', '/')
            match = annotations_df[annotations_df['ImagePath'] == img_path_standard]
            if not match.empty:
                print(f"Trovata corrispondenza per {img} nel CSV")
            else:
                print(f"Nessuna corrispondenza per {img} nel CSV")
    else:
        print(f"ATTENZIONE: Cartella {mulo_dir} non trovata")

    # Raccolta immagini dalla cartella NonMulo
    nonmulo_dir = os.path.join(DATASET_DIR, 'NonMulo')
    if os.path.exists(nonmulo_dir):
        nonmulo_images = [os.path.join('NonMulo', img) for img in os.listdir(nonmulo_dir) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
        print(f"Trovate {len(nonmulo_images)} immagini nella cartella NonMulo")
        all_images.extend([(img, 0) for img in nonmulo_images])  # 0 = non è un mulo
    else:
        print(f"ATTENZIONE: Cartella {nonmulo_dir} non trovata")

    # Shuffle e divisione train/val
    random.seed(42)  # Per riproducibilità
    random.shuffle(all_images)
    split_idx = int(len(all_images) * 0.8)  # 80% training, 20% validation
    train_images = all_images[:split_idx]
    val_images = all_images[split_idx:]

    print(f"Dataset diviso in {len(train_images)} immagini di training e {len(val_images)} di validazione")

    # Prepara le immagini per il training
    for img_path, is_mulo in train_images:
        prepare_image_for_yolo(img_path, is_mulo, "train")

    # Prepara le immagini per la validazione
    for img_path, is_mulo in val_images:
        prepare_image_for_yolo(img_path, is_mulo, "val")

    # Crea il file YAML per la configurazione del dataset
    yaml_content = f"""
path: {os.path.abspath(OUTPUT_DIR)}
train: images/train
val: images/val

# Classi
names:
  0: mulo
"""

    with open(os.path.join(OUTPUT_DIR, 'dataset.yaml'), 'w') as f:
        f.write(yaml_content)

    print("Preparazione dataset completata!")

    # ----------------- ADDESTRAMENTO -----------------
    print("Inizio addestramento del modello...")

    # Carica un modello pre-addestrato YOLOv8 (versione piccola per iniziare)
    model = YOLO('yolov8n.pt')

    # Addestra il modello
    results = model.train(
        data=os.path.join(OUTPUT_DIR, 'dataset.yaml'),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        patience=PATIENCE,
        verbose=True,
        pretrained=True,
        lr0=0.01,  # Learning rate iniziale più alto per fine-tuning
        lrf=0.001,  # Learning rate finale
        save=True,  # Salva il modello migliore
        plots=True,  # Genera grafici di addestramento
        device='0',  # Usa GPU 0
        workers=0    # Disattiva il multiprocessing per evitare problemi su Windows
    )

    # ----------------- ESPORTAZIONE -----------------
    print("Esportazione del modello in formato ONNX...")

    # Esporta il modello in formato ONNX (ottimizzato per ML.NET)
    model.export(format='onnx', imgsz=IMG_SIZE)
    print(f"Modello esportato: {model.export_filename}")

    # ----------------- VALUTAZIONE -----------------
    print("Valutazione del modello...")

    # Valuta il modello
    val_results = model.val()
    print(f"mAP50-95: {val_results.box.map}")
    print(f"Precision: {val_results.box.mp}")
    print(f"Recall: {val_results.box.mr}")

    print("Addestramento completato!")

if __name__ == '__main__':
    # Necessario per Windows
    multiprocessing.freeze_support()
    
    # Verifica CUDA
    print(f"PyTorch versione: {torch.__version__}")
    print(f"CUDA disponibile: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"Dispositivo CUDA: {torch.cuda.get_device_name(0)}")
        print(f"Numero di GPU disponibili: {torch.cuda.device_count()}")
        
        # Testa un modello base su GPU
        model = YOLO('yolov8n.pt')
        print(f"Dispositivo modello: {model.device}")
        
        # Forza l'uso della GPU
        if str(model.device) == 'cpu':
            print("Forzatura del dispositivo su GPU...")
            model.to('cuda:0')
            print(f"Nuovo dispositivo modello: {model.device}")
    else:
        print("CUDA non disponibile. L'addestramento userà solo CPU.")
    
    # Esegui il codice principale
    main()