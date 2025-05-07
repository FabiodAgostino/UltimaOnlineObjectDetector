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
DATASET_DIR = 'Dataset'  # Cartella principale del dataset
OUTPUT_DIR = 'YoloDataset'  # Cartella di output per il dataset in formato YOLO
EXISTING_MODEL = 'best.pt'  # Percorso del modello esistente (modello mulo)

# Parametri di addestramento
IMG_SIZE = 640
EPOCHS = 100
BATCH_SIZE = 16
PATIENCE = 20

# Funzione per ottenere le dimensioni dell'immagine
def get_image_dimensions(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return 640, 640  # Valori predefiniti se l'immagine non può essere letta
    return img.shape[1], img.shape[0]  # width, height

# Funzione per preparare un'immagine per YOLO
def prepare_image_for_yolo(img_path, class_id, subset="train"):
    # Ottieni il percorso completo dell'immagine
    full_img_path = os.path.join(DATASET_DIR, img_path)
    img_name = os.path.basename(img_path)
    
    # Crea cartelle se non esistono
    os.makedirs(os.path.join(OUTPUT_DIR, 'images', subset), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, 'labels', subset), exist_ok=True)
    
    # Copia l'immagine nella directory corretta
    dest_path = os.path.join(OUTPUT_DIR, 'images', subset, img_name)
    try:
        shutil.copy(full_img_path, dest_path)
        print(f"Copiato {full_img_path} in {dest_path}")
    except Exception as e:
        print(f"Errore nel copiare {full_img_path}: {e}")
        return
    
    # Cerca nel DataFrame le annotazioni
    img_row = annotations_df[annotations_df['ImagePath'] == img_path]
    
    # Ottieni le dimensioni dell'immagine
    img_width, img_height = get_image_dimensions(full_img_path)
    
    # Prepara le coordinate per l'annotazione YOLO
    if not img_row.empty and 'X' in annotations_df.columns and 'Y' in annotations_df.columns:
        # Usa le coordinate annotate
        x_center = float(img_row['X'].values[0]) / img_width
        y_center = float(img_row['Y'].values[0]) / img_height
        
        # Verifica se il punto è un mulo (1) o non-mulo (0)
        if 'IsMulo' in annotations_df.columns:
            # Sovrascrive class_id con il valore dal CSV
            is_mulo = int(img_row['IsMulo'].values[0])
            class_id = 0 if is_mulo == 1 else 1  # Assicura che 1=mulo corrisponda a classe 0
    else:
        # Default: centro dell'immagine se non ci sono annotazioni
        x_center = 0.5
        y_center = 0.5
    
    # Per YOLO, width e height sono stimati 
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

def main():
    global annotations_df  # Rendiamo annotations_df globale
    
    # ----------------- PREPARAZIONE DATASET -----------------
    print("Preparazione del dataset...")

    # Crea le cartelle necessarie
    os.makedirs(os.path.join(OUTPUT_DIR, 'images', 'train'), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, 'images', 'val'), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, 'labels', 'train'), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, 'labels', 'val'), exist_ok=True)

    # Leggi il file CSV con le annotazioni
    csv_path = os.path.join(DATASET_DIR, 'dataset.csv')
    if os.path.exists(csv_path):
        try:
            # Tenta di caricare il CSV con intestazione
            annotations_df = pd.read_csv(csv_path)
            print(f"File CSV caricato con {len(annotations_df)} annotazioni e intestazioni")
        except Exception:
            # Se fallisce, prova senza intestazione
            annotations_df = pd.read_csv(csv_path, header=None, 
                                         names=['ImagePath', 'IsMulo', 'X', 'Y'])
            print(f"File CSV caricato con {len(annotations_df)} annotazioni senza intestazioni")
        
        # Stampa le prime righe e le colonne per debug
        print("Prime righe del CSV:")
        print(annotations_df.head())
        print("Colonne del CSV:", annotations_df.columns.tolist())
    else:
        print(f"File CSV di annotazioni non trovato in {csv_path}, creando dataset senza coordinate specifiche")
        annotations_df = pd.DataFrame(columns=['ImagePath', 'IsMulo', 'X', 'Y'])

    # Raccogli tutte le immagini dalla cartella Mulo
    mulo_images = []
    mulo_dir = os.path.join(DATASET_DIR, 'Mulo')
    
    if os.path.exists(mulo_dir):
        mulo_files = [f for f in os.listdir(mulo_dir) 
                     if os.path.isfile(os.path.join(mulo_dir, f)) and 
                     f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        print(f"Trovate {len(mulo_files)} immagini nella cartella Mulo")
        
        # Crea percorsi relativi per le immagini
        mulo_images = [(os.path.join('Mulo', img), 0) for img in mulo_files]  # Classe 0 = mulo
    else:
        print(f"ATTENZIONE: Cartella {mulo_dir} non trovata")

    # Raccogli immagini dalla cartella Lama (se esiste e ha immagini)
    lama_dir = os.path.join(DATASET_DIR, 'Lama')
    if os.path.exists(lama_dir):
        lama_files = [f for f in os.listdir(lama_dir) 
                     if os.path.isfile(os.path.join(lama_dir, f)) and 
                     f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        print(f"Trovate {len(lama_files)} immagini nella cartella Lama")
        
        # Aggiungi percorsi relativi per le immagini di lama
        lama_images = [(os.path.join('Lama', img), 1) for img in lama_files]  # Classe 1 = lama
        mulo_images.extend(lama_images)

    # Shuffle e divisione train/val
    if mulo_images:
        random.seed(42)  # Per riproducibilità
        random.shuffle(mulo_images)
        split_idx = int(len(mulo_images) * 0.8)  # 80% training, 20% validation
        train_images = mulo_images[:split_idx]
        val_images = mulo_images[split_idx:]

        print(f"Dataset diviso in {len(train_images)} immagini di training e {len(val_images)} di validazione")

        # Prepara le immagini per il training e la validazione
        for img_path, class_id in train_images:
            prepare_image_for_yolo(img_path, class_id, "train")
        
        for img_path, class_id in val_images:
            prepare_image_for_yolo(img_path, class_id, "val")
    else:
        print("ERRORE: Nessuna immagine trovata nelle cartelle specificate!")
        return

    # Crea il file YAML per la configurazione del dataset
    yaml_content = """path: {}
train: images/train
val: images/val

# Classi
names:
  0: mulo
""".format(os.path.abspath(OUTPUT_DIR))

    yaml_path = os.path.join(OUTPUT_DIR, 'dataset.yaml')
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    print(f"File di configurazione YAML creato in {yaml_path}")
    print("Contenuto del file YAML:")
    print(yaml_content)
    print("Preparazione dataset completata!")

    # ----------------- ADDESTRAMENTO -----------------
    print("Inizio addestramento del modello...")

    # Verifica se esiste il modello precedente
    if os.path.exists(EXISTING_MODEL):
        print(f"Utilizzo del modello esistente {EXISTING_MODEL} per transfer learning")
        model = YOLO(EXISTING_MODEL)
    else:
        print("Modello esistente non trovato, inizializzo un nuovo modello YOLOv8n")
        model = YOLO('yolov8n.pt')

    # Addestra il modello
    results = model.train(
        data=yaml_path,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        patience=PATIENCE,
        verbose=True,
        pretrained=True,
        lr0=0.01,
        lrf=0.001,
        save=True,
        plots=True,
        device='0' if torch.cuda.is_available() else 'cpu',
        workers=0  # Disattiva il multiprocessing per evitare problemi su Windows
    )

    # ----------------- ESPORTAZIONE -----------------
    print("Esportazione del modello in formato ONNX...")

    # Esporta il modello in formato ONNX
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
    else:
        print("CUDA non disponibile. L'addestramento userà solo CPU.")
    
    # Esegui il codice principale
    main()