import os
import sys
import numpy as np
from PIL import Image
from ultralytics import YOLO

def detect_object(image_path, model_path="best.pt", confidence_threshold=0.5):
    """
    Analizza un'immagine per rilevare oggetti (muli o lama).
    
    Args:
        image_path: Percorso dell'immagine da analizzare
        model_path: Percorso del modello YOLO addestrato
        confidence_threshold: Soglia di confidenza minima per il rilevamento
        
    Returns:
        Una tupla (class_id, x, y) con la classe dell'oggetto e le coordinate se trovato, None altrimenti
    """
    try:
        # Verifica che il file immagine esista
        if not os.path.exists(image_path):
            print(f"Errore: Il file {image_path} non esiste.")
            return None
            
        # Verifica che il modello esista
        if not os.path.exists(model_path):
            print(f"Errore: Il modello {model_path} non esiste.")
            return None
            
        # Carica il modello YOLO
        model = YOLO(model_path)
        
        # Esegui il rilevamento
        results = model(image_path)
        
        # Controlla se ci sono rilevamenti
        if len(results) > 0 and len(results[0].boxes) > 0:
            # Ottieni confidenze dei rilevamenti
            boxes = results[0].boxes
            confidences = boxes.conf.cpu().numpy()
            
            # Se nessuna confidenza supera la soglia, restituisci None
            if len(confidences) == 0 or max(confidences) < confidence_threshold:
                return None
                
            # Trova l'indice del rilevamento con la confidenza più alta
            best_idx = np.argmax(confidences)
            
            # Ottieni la classe del rilevamento
            if hasattr(boxes, 'cls') and len(boxes.cls) > 0:
                class_id = int(boxes.cls[best_idx].item())
            else:
                class_id = 0  # Default a mulo se non ci sono informazioni sulla classe
            
            # Ottieni le coordinate della bounding box (x1, y1, x2, y2)
            box = boxes.xyxy.cpu().numpy()[best_idx]
            x1, y1, x2, y2 = box
            
            # Calcola il centro del rilevamento
            x_center = int((x1 + x2) / 2)
            y_center = int((y1 + y2) / 2)
            
            # Restituisci la classe e le coordinate
            return (class_id, x_center, y_center)
        else:
            return None
    
    except Exception as e:
        print(f"Errore durante il rilevamento: {e}")
        return None

if __name__ == "__main__":
    # Controlla se è stato fornito un argomento (percorso immagine)
    if len(sys.argv) < 2:
        print("Utilizzo: python detect_object.py <percorso_immagine>")
        sys.exit(1)
        
    image_path = sys.argv[1]
    result = detect_object(image_path)
    
    if result:
        class_id, x, y = result
        class_name = "mulo" if class_id == 0 else "lama" if class_id == 1 else f"classe_{class_id}"
        print(f"{class_name},{x},{y}")  # Output semplice per facile parsing
    else:
        print("null")  # Output "null" se non viene rilevato alcun oggetto