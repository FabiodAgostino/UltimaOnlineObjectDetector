import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import json
from PIL import Image, ImageDraw, ImageFont
import cv2

class MuloDetector:
    def __init__(self, dataset_dir="Dataset", output_dir="Result", confidence_threshold=0.30):
        """
        Inizializza il rilevatore di muli con YOLO.
        
        Args:
            dataset_dir: Directory contenente le immagini da analizzare
            output_dir: Directory dove verranno salvati i risultati
            confidence_threshold: Soglia minima di confidenza per considerare valido un rilevamento
        """
        self.dataset_dir = dataset_dir
        self.output_dir = output_dir
        self.mulo_dir = os.path.join(dataset_dir, "Mulo")
        self.nonmulo_dir = os.path.join(dataset_dir, "NonMulo")
        self.results_dir = os.path.join(output_dir, "images")
        self.confidence_threshold = confidence_threshold
        
        self.stats = {
            "total_images": 0,
            "mulo_images": 0,
            "nonmulo_images": 0,
            "processed_images": 0,
            "success_rate": 0,
            "failure_rate": 0,
            "mulo_detected": 0,
            "mulo_not_detected": 0,
            "false_positives": 0,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "details": [],
            "confidence_values": []  # Nuovo campo per registrare tutti i valori di confidenza
        }
        
        # Crea le directory di output
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Carica il modello YOLO
        try:
            from ultralytics import YOLO
            model_path = "best.pt"
            
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Modello YOLO non trovato in {model_path}")
            
            print(f"Caricamento del modello YOLO da {model_path}...")
            self.yolo_model = YOLO(model_path)
            
            # Imposta esplicitamente la soglia di confidenza
            self.yolo_model.conf = self.confidence_threshold
            
            print(f"Modello YOLO caricato con successo!")
            print(f"Parametri del modello: {self.yolo_model.model.names}")  # Mostra i nomi delle classi
            print(f"Soglia di confidenza impostata a: {self.confidence_threshold}")
            
        except ImportError:
            raise ImportError("La libreria ultralytics (YOLO) non è installata. Installala con 'pip install ultralytics'")
    
    def detect_mulo(self, image_path):
        """
        Rileva se l'immagine contiene un mulo utilizzando il modello YOLO.
        
        Args:
            image_path: Percorso dell'immagine da analizzare
            
        Returns:
            Tupla (x_center, y_center, detection_method, is_mulo_detected, is_in_mulo_folder)
        """
        img_name = os.path.basename(image_path)
        is_in_mulo_folder = "Mulo" == os.path.basename(os.path.dirname(image_path))
        
        # Verifica che l'immagine esista
        if not os.path.exists(image_path):
            print(f"ERRORE: Immagine non trovata: {image_path}")
            return None, None, "file_not_found", False, is_in_mulo_folder
            
        # Verifica dimensioni immagine prima dell'inferenza
        try:
            with Image.open(image_path) as img:
                width, height = img.size
                print(f"DEBUG - Immagine: {img_name}, Dimensioni: {width}x{height}, Categoria: {'Mulo' if is_in_mulo_folder else 'NonMulo'}")
        except Exception as e:
            print(f"ERRORE nell'apertura dell'immagine {image_path}: {e}")
            return None, None, "invalid_image", False, is_in_mulo_folder
        
        try:
            # Esegui rilevamento con YOLO
            results = self.yolo_model(image_path)
            
            # Informazioni di debug
            print(f"DEBUG - Rilevamenti per {img_name}: {len(results[0].boxes) if len(results) > 0 else 0}")
            
            # Elabora i risultati
            if len(results) > 0 and len(results[0].boxes) > 0:
                # Prendi tutti i rilevamenti
                boxes = results[0].boxes
                confidences = boxes.conf.cpu().numpy()
                
                # Informazioni di debug
                print(f"DEBUG - Confidenze: {confidences}")
                
                # Se non ci sono rilevamenti con confidenza sufficiente
                if len(confidences) == 0 or max(confidences) < self.confidence_threshold:
                    print(f"DEBUG - Nessun rilevamento sopra la soglia ({self.confidence_threshold}) per {img_name}")
                    return None, None, "yolo_no_detection", False, is_in_mulo_folder
                    
                # Trova l'indice del rilevamento con la confidenza più alta
                best_idx = np.argmax(confidences)
                confidence = confidences[best_idx]
                
                # Registra il valore di confidenza per analisi
                self.stats["confidence_values"].append(float(confidence))
                
                # Ottieni le coordinate della bounding box (x1, y1, x2, y2)
                box = boxes.xyxy.cpu().numpy()[best_idx]
                x1, y1, x2, y2 = box
                
                # Calcola il centro del rilevamento
                x_center = int((x1 + x2) / 2)
                y_center = int((y1 + y2) / 2)
                
                print(f"DEBUG - Mulo rilevato in {img_name} con confidenza {confidence:.2f} alle coordinate ({x_center}, {y_center})")
                return x_center, y_center, f"yolo_detection_{confidence:.2f}", True, is_in_mulo_folder
            else:
                print(f"DEBUG - Nessun rilevamento per {img_name}")
                return None, None, "yolo_no_detection", False, is_in_mulo_folder
        except Exception as e:
            print(f"ERRORE nel rilevamento YOLO per {image_path}: {e}")
            return None, None, f"yolo_error: {str(e)}", False, is_in_mulo_folder
    
    def process_image(self, image_path):
        """
        Elabora un'immagine e disegna un punto al centro del mulo se rilevato.
        
        Args:
            image_path: Percorso dell'immagine da elaborare
            
        Returns:
            Boolean che indica se l'elaborazione è avvenuta con successo
        """
        img_name = os.path.basename(image_path)
        
        # Rileva il mulo
        x_center, y_center, detection_method, is_mulo_detected, is_in_mulo_folder = self.detect_mulo(image_path)
        
        if x_center is None or y_center is None:
            # Aggiorna le statistiche per immagine non elaborabile
            self.stats["processed_images"] += 1
            if is_in_mulo_folder:
                self.stats["mulo_not_detected"] += 1
            
            self.stats["details"].append({
                "image": img_name,
                "category": "Mulo" if is_in_mulo_folder else "NonMulo",
                "status": "failed",
                "detection_method": detection_method,
                "coordinates": None
            })
            
            # Copia l'immagine originale con etichetta di errore
            try:
                img = Image.open(image_path)
                draw = ImageDraw.Draw(img)
                
                # Etichette corrette
                source_label = "IsMulo" if is_in_mulo_folder else "NonMulo"
                detection_label = "Nessun rilevamento YOLO"
                
                try:
                    # Prova a usare un font di sistema
                    font = ImageFont.truetype("arial.ttf", 20)
                    draw.text((10, 10), source_label, fill=(0, 0, 255), font=font)  # Blu per la categoria
                    draw.text((10, 40), detection_label, fill=(255, 0, 0), font=font)  # Rosso per il non rilevamento
                    draw.text((10, 70), detection_method, fill=(255, 165, 0), font=font)  # Arancione per il metodo
                except:
                    # Se il font non è disponibile, usa il font di default
                    draw.text((10, 10), source_label, fill=(0, 0, 255))
                    draw.text((10, 40), detection_label, fill=(255, 0, 0))
                    draw.text((10, 70), detection_method, fill=(255, 165, 0))
                
                # Salva l'immagine risultante
                output_path = os.path.join(self.results_dir, img_name)
                img.save(output_path)
            except Exception as e:
                print(f"ERRORE nel salvare l'immagine di errore per {image_path}: {e}")
            
            return False
        
        # Carica l'immagine per disegnare il punto
        try:
            img = Image.open(image_path)
            draw = ImageDraw.Draw(img)
            
            # Disegna il punto rosso SOLO se è un mulo rilevato
            if is_mulo_detected:
                point_radius = 5
                draw.ellipse((x_center-point_radius, y_center-point_radius, 
                            x_center+point_radius, y_center+point_radius), 
                            fill=(255, 0, 0))  # Punto rosso
                draw.ellipse((x_center-point_radius-2, y_center-point_radius-2, 
                            x_center+point_radius+2, y_center+point_radius+2), 
                            outline=(255, 255, 255), width=2)  # Bordo bianco
            
            # Etichette corrette
            source_label = "IsMulo" if is_in_mulo_folder else "NonMulo"
            detection_label = "Mulo" if is_mulo_detected else "NonMulo"
            
            try:
                # Prova a usare un font di sistema - se non disponibile, salta questa parte
                font = ImageFont.truetype("arial.ttf", 20)
                # Posiziona le etichette in punti diversi
                draw.text((10, 10), source_label, fill=(0, 0, 255), font=font)  # Blu per la categoria
                draw.text((10, 40), detection_label, fill=(0, 255, 0), font=font)  # Verde per il rilevamento
                # Aggiungi il metodo di rilevamento come terza etichetta
                draw.text((10, 70), detection_method, fill=(255, 165, 0), font=font)  # Arancione per il metodo
            except:
                # Se il font non è disponibile, usa il font di default
                draw.text((10, 10), source_label, fill=(0, 0, 255))
                draw.text((10, 40), detection_label, fill=(0, 255, 0))
                draw.text((10, 70), detection_method, fill=(255, 165, 0))
            
            # Salva l'immagine risultante
            output_path = os.path.join(self.results_dir, img_name)
            img.save(output_path)
            
        except Exception as e:
            print(f"ERRORE nell'elaborazione dell'immagine {image_path}: {e}")
            return False
        
        # Aggiorna le statistiche
        self.stats["processed_images"] += 1
        
        # Statistiche corrette basate sulla categoria e il rilevamento
        if is_in_mulo_folder and is_mulo_detected:
            self.stats["mulo_detected"] += 1
        elif is_in_mulo_folder and not is_mulo_detected:
            self.stats["mulo_not_detected"] += 1
        elif not is_in_mulo_folder and is_mulo_detected:
            self.stats["false_positives"] += 1
        
        # Aggiungi dettagli per questa immagine
        self.stats["details"].append({
            "image": img_name,
            "category": "Mulo" if is_in_mulo_folder else "NonMulo",
            "status": "success" if ((is_in_mulo_folder and is_mulo_detected) or 
                                   (not is_in_mulo_folder and not is_mulo_detected)) else "failed",
            "detection_method": detection_method,
            "coordinates": (x_center, y_center) if x_center is not None else None
        })
        
        return True
    
    def process_all_images(self):
        """
        Elabora tutte le immagini nelle cartelle Mulo e NonMulo.
        """
        # Verifica l'esistenza delle cartelle prima di elencare i file
        if not os.path.exists(self.mulo_dir):
            print(f"ATTENZIONE: Cartella {self.mulo_dir} non trovata!")
            mulo_images = []
        else:
            mulo_images = [os.path.join(self.mulo_dir, img) for img in os.listdir(self.mulo_dir) 
                          if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if not os.path.exists(self.nonmulo_dir):
            print(f"ATTENZIONE: Cartella {self.nonmulo_dir} non trovata!")
            nonmulo_images = []
        else:
            nonmulo_images = [os.path.join(self.nonmulo_dir, img) for img in os.listdir(self.nonmulo_dir) 
                             if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        self.stats["mulo_images"] = len(mulo_images)
        self.stats["nonmulo_images"] = len(nonmulo_images)
        self.stats["total_images"] = self.stats["mulo_images"] + self.stats["nonmulo_images"]
        
        # Elabora ogni immagine
        print(f"Elaborazione di {self.stats['mulo_images']} immagini di muli...")
        for idx, img_path in enumerate(mulo_images):
            print(f"[{idx+1}/{self.stats['mulo_images']}] Elaborazione di {os.path.basename(img_path)}")
            self.process_image(img_path)
        
        print(f"Elaborazione di {self.stats['nonmulo_images']} immagini di non-muli...")
        for idx, img_path in enumerate(nonmulo_images):
            print(f"[{idx+1}/{self.stats['nonmulo_images']}] Elaborazione di {os.path.basename(img_path)}")
            self.process_image(img_path)
    
    def generate_report(self):
        """
        Genera un report dettagliato delle statistiche di elaborazione.
        """
        # Calcola tassi finali
        if self.stats["mulo_images"] > 0:
            self.stats["success_rate"] = (self.stats["mulo_detected"] / self.stats["mulo_images"]) * 100
            self.stats["failure_rate"] = 100 - self.stats["success_rate"]
        
        # Genera report testuale
        report_path = os.path.join(self.output_dir, "report.txt")
        with open(report_path, "w") as f:
            f.write("=== REPORT RILEVAMENTO MULI CON YOLO ===\n")
            f.write(f"Data e ora: {self.stats['timestamp']}\n\n")
            f.write(f"Soglia di confidenza: {self.confidence_threshold}\n\n")
            f.write(f"Totale immagini elaborate: {self.stats['total_images']}\n")
            f.write(f"- Immagini di muli: {self.stats['mulo_images']}\n")
            f.write(f"- Immagini non-muli: {self.stats['nonmulo_images']}\n\n")
            f.write(f"Muli rilevati correttamente: {self.stats['mulo_detected']}\n")
            f.write(f"Muli non rilevati: {self.stats['mulo_not_detected']}\n")
            f.write(f"Falsi positivi: {self.stats['false_positives']}\n\n")
            f.write(f"Tasso di successo: {self.stats['success_rate']:.2f}%\n")
            f.write(f"Tasso di fallimento: {self.stats['failure_rate']:.2f}%\n\n")
            f.write("=== DETTAGLI ELABORAZIONE ===\n")
            
            for detail in self.stats["details"]:
                f.write(f"Immagine: {detail['image']}\n")
                f.write(f"  Categoria: {detail['category']}\n")
                f.write(f"  Stato: {detail['status']}\n")
                f.write(f"  Metodo di rilevamento: {detail['detection_method']}\n")
                f.write(f"  Coordinate: {detail['coordinates']}\n\n")
        
        # Salva anche come JSON per un'analisi più approfondita
        json_path = os.path.join(self.output_dir, "report.json")
        with open(json_path, "w") as f:
            json.dump(self.stats, f, indent=2)
        
        # Genera grafici
        self.generate_charts()
        
        print(f"Report generato in: {report_path}")
        print(f"Dati JSON salvati in: {json_path}")
    
    def generate_charts(self):
        """
        Genera grafici per visualizzare le statistiche.
        """
        # Grafico a torta per il tasso di successo/fallimento
        if self.stats["mulo_images"] > 0:  # Previeni divisione per zero
            plt.figure(figsize=(10, 6))
            labels = ["Rilevati", "Non rilevati"]
            sizes = [self.stats["mulo_detected"], self.stats["mulo_not_detected"]]
            colors = ["#4CAF50", "#F44336"]
            plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
            plt.axis('equal')
            plt.title(f"Tasso di rilevamento muli con YOLO (soglia: {self.confidence_threshold})")
            plt.savefig(os.path.join(self.output_dir, "detection_rate.png"))
            plt.close()
        
        # Grafico a barre per le categorie di immagini
        plt.figure(figsize=(12, 6))
        categories = ["Totale", "Muli", "Non-Muli", "Rilevati", "Non rilevati", "Falsi positivi"]
        values = [
            self.stats["total_images"],
            self.stats["mulo_images"],
            self.stats["nonmulo_images"],
            self.stats["mulo_detected"],
            self.stats["mulo_not_detected"],
            self.stats["false_positives"]
        ]
        colors = ["#2196F3", "#4CAF50", "#FFC107", "#9C27B0", "#F44336", "#FF9800"]
        
        plt.bar(categories, values, color=colors)
        plt.ylabel("Numero di immagini")
        plt.title(f"Statistiche di elaborazione con YOLO (soglia: {self.confidence_threshold})")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "statistics.png"))
        plt.close()
        
        # Istogramma delle confidenze di rilevamento
        if self.stats["confidence_values"]:
            plt.figure(figsize=(10, 6))
            plt.hist(self.stats["confidence_values"], bins=20, color="#00BCD4", alpha=0.7, edgecolor='black')
            plt.axvline(x=self.confidence_threshold, color='r', linestyle='--', label=f'Soglia ({self.confidence_threshold})')
            plt.xlabel("Confidenza")
            plt.ylabel("Numero di rilevamenti")
            plt.title("Distribuzione delle confidenze YOLO")
            plt.grid(axis='y', alpha=0.75)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, "confidence_distribution.png"))
            plt.close()
        
        # Grafico a barre delle confidenze per ogni immagine rilevata
        detections = [detail for detail in self.stats["details"] 
                     if detail["status"] == "success" and detail["category"] == "Mulo"]
        
        if detections:
            plt.figure(figsize=(14, 8))
            image_names = [d["image"] for d in detections]
            confidences = []
            
            for d in detections:
                method = d["detection_method"]
                if "yolo_detection_" in method:
                    conf = float(method.replace("yolo_detection_", ""))
                    confidences.append(conf)
                else:
                    confidences.append(0)
            
            # Ordina per confidenza
            sorted_indices = np.argsort(confidences)[::-1]  # Ordine decrescente
            image_names = [image_names[i] for i in sorted_indices]
            confidences = [confidences[i] for i in sorted_indices]
            
            plt.figure(figsize=(14, 8))
            plt.bar(range(len(confidences)), confidences, color="#2196F3")
            plt.axhline(y=self.confidence_threshold, color='r', linestyle='--', label=f'Soglia ({self.confidence_threshold})')
            plt.xlabel("Immagine")
            plt.ylabel("Confidenza")
            plt.title("Confidenza per ogni mulo rilevato (ordinato)")
            plt.xticks(range(len(confidences)), image_names, rotation=90)
            plt.tight_layout()
            plt.grid(axis='y', alpha=0.3)
            plt.legend()
            plt.savefig(os.path.join(self.output_dir, "confidence_by_image.png"))
            plt.close()
    
    def test_specific_images(self, image_paths):
        """
        Testa il modello su immagini specifiche e mostra i risultati dettagliati.
        
        Args:
            image_paths: Lista di percorsi delle immagini da testare
        """
        print("\n=== TEST DIRETTO SU IMMAGINI SPECIFICHE ===")
        
        for img_path in image_paths:
            try:
                print(f"\nTest diretto su: {img_path}")
                if not os.path.exists(img_path):
                    print(f"ERRORE: File non trovato: {img_path}")
                    continue
                
                # Test con differenti soglie di confidenza
                original_conf = self.yolo_model.conf
                
                test_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
                for threshold in test_thresholds:
                    self.yolo_model.conf = threshold
                    print(f"\nTest con soglia di confidenza: {threshold}")
                    
                    results = self.yolo_model(img_path)
                    
                    if len(results[0].boxes) > 0:
                        boxes = results[0].boxes
                        confidences = boxes.conf.cpu().numpy()
                        print(f"Rilevamenti: {len(confidences)}")
                        print(f"Confidenze: {confidences}")
                        
                        # Mostra il miglior rilevamento
                        if len(confidences) > 0:
                            best_idx = np.argmax(confidences)
                            confidence = confidences[best_idx]
                            box = boxes.xyxy.cpu().numpy()[best_idx]
                            x1, y1, x2, y2 = box
                            x_center = int((x1 + x2) / 2)
                            y_center = int((y1 + y2) / 2)
                            print(f"Miglior rilevamento: confidenza {confidence:.2f}, centro ({x_center}, {y_center})")
                            
                            # Salva l'immagine con il riquadro di rilevamento
                            img = cv2.imread(img_path)
                            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                            cv2.circle(img, (x_center, y_center), 5, (0, 0, 255), -1)
                            cv2.putText(img, f"{confidence:.2f}", (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                            test_output_dir = os.path.join(self.output_dir, "test_images")
                            os.makedirs(test_output_dir, exist_ok=True)
                            output_path = os.path.join(test_output_dir, f"{os.path.basename(img_path).split('.')[0]}_conf{threshold:.1f}.jpg")
                            cv2.imwrite(output_path, img)
                            print(f"Immagine salvata in: {output_path}")
                    else:
                        print(f"Nessun rilevamento con soglia {threshold}")
                
                # Ripristina la soglia originale
                self.yolo_model.conf = original_conf
                
            except Exception as e:
                print(f"ERRORE nel test dell'immagine {img_path}: {e}")
                
    def test_data_samples(self, num_samples=5):
        """
        Testa il modello su un campione casuale di immagini dal dataset.
        
        Args:
            num_samples: Numero di immagini di esempio da testare da ogni categoria
        """
        # Ottieni un campione di immagini di muli
        mulo_images = []
        if os.path.exists(self.mulo_dir):
            all_mulo = [os.path.join(self.mulo_dir, img) for img in os.listdir(self.mulo_dir) 
                       if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if len(all_mulo) > 0:
                mulo_images = np.random.choice(all_mulo, min(num_samples, len(all_mulo)), replace=False)
        
        # Ottieni un campione di immagini di non-muli
        nonmulo_images = []
        if os.path.exists(self.nonmulo_dir):
            all_nonmulo = [os.path.join(self.nonmulo_dir, img) for img in os.listdir(self.nonmulo_dir) 
                          if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if len(all_nonmulo) > 0:
                nonmulo_images = np.random.choice(all_nonmulo, min(num_samples, len(all_nonmulo)), replace=False)
        
        # Testa i campioni
        test_images = list(mulo_images) + list(nonmulo_images)
        if test_images:
            self.test_specific_images(test_images)
        else:
            print("Nessuna immagine di test disponibile.")


def main():
    """
    Funzione principale per l'esecuzione del programma.
    """
    print("=== RILEVATORE DI MULI con YOLO ===")
    print("Inizializzazione...")
    
    # Parametri impostabili
    confidence_threshold = 0.30  # Soglia di confidenza personalizzabile
    dataset_dir = "Dataset"
    output_dir = "Result"
    
    # Verifica se le cartelle esistono
    if not os.path.exists(dataset_dir):
        print(f"ERRORE: Cartella '{dataset_dir}' non trovata!")
        return
    
    if not os.path.exists(os.path.join(dataset_dir, "Mulo")) and not os.path.exists(os.path.join(dataset_dir, "NonMulo")):
        print(f"ERRORE: Nessuna cartella di immagini trovata in '{dataset_dir}'!")
        return
    
    try:
        # Crea il rilevatore di muli
        detector = MuloDetector(dataset_dir=dataset_dir, output_dir=output_dir, confidence_threshold=confidence_threshold)
        
        # Test opzionale su alcuni campioni del dataset
        detector.test_data_samples(5)
        
        print("\nAvvio elaborazione completa delle immagini...")
        detector.process_all_images()
        
        print("Generazione report...")
        detector.generate_report()
        
        print("Elaborazione completata con successo!")
        print(f"Immagini elaborate: {detector.stats['processed_images']}")
        print(f"Muli rilevati: {detector.stats['mulo_detected']}/{detector.stats['mulo_images']}")
        if detector.stats["mulo_images"] > 0:
            print(f"Tasso di successo: {detector.stats['success_rate']:.2f}%")
        print(f"Risultati salvati nella cartella '{output_dir}'")
    except Exception as e:
        print(f"ERRORE critico: {e}")
        import traceback
        traceback.print_exc()
        print("L'elaborazione è stata interrotta.")


if __name__ == "__main__":
    main()