using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace UltimaOnlineObjectDetector
{
    public class YoloDetector : IDisposable
    {
        private readonly InferenceSession _session;
        private readonly float _confidenceThreshold;
        private readonly int _inputWidth;
        private readonly int _inputHeight;
        private bool _disposed = false;

        /// <summary>
        /// Inizializza un nuovo detector di muli utilizzando un modello ONNX.
        /// </summary>
        public YoloDetector(string modelPath, float confidenceThreshold = 0.25f, int inputWidth = 640, int inputHeight = 640)
        {
            if (!File.Exists(modelPath))
            {
                throw new FileNotFoundException($"Il modello ONNX non è stato trovato: {modelPath}");
            }

            _confidenceThreshold = confidenceThreshold;
            _inputWidth = inputWidth;
            _inputHeight = inputHeight;

            try
            {
                // Crea una sessione di inferenza ONNX
                var sessionOptions = new SessionOptions();
                sessionOptions.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;

                _session = new InferenceSession(modelPath, sessionOptions);
            }
            catch (Exception ex)
            {
                throw new Exception($"Errore nel caricamento del modello ONNX: {ex.Message}", ex);
            }
        }

        /// <summary>
        /// Rileva se l'immagine contiene un mulo e restituisce le coordinate del centro.
        /// </summary>
        public (int X, int Y)? DetectMulo(string imagePath)
        {
            if (!File.Exists(imagePath))
            {
                throw new FileNotFoundException($"L'immagine non è stata trovata: {imagePath}");
            }

            try
            {
                // Carica l'immagine originale
                using var originalImage = Image.Load<Rgb24>(imagePath);
                int originalWidth = originalImage.Width;
                int originalHeight = originalImage.Height;

                // Prepara l'immagine per il modello
                DenseTensor<float> imageData = PrepareImage(imagePath, out var scaleInfo);

                // Ottieni il nome del primo input del modello
                string inputName = _session.InputMetadata.Keys.First();

                // Esegui l'inferenza
                var inputs = new List<NamedOnnxValue>
                {
                    NamedOnnxValue.CreateFromTensor(inputName, imageData)
                };

                // Misura il tempo di inferenza
                var watch = System.Diagnostics.Stopwatch.StartNew();
                using var results = _session.Run(inputs);
                watch.Stop();

                // Estrai i risultati
                var detections = new List<Detection>();

                foreach (var result in results)
                {
                    var outputTensor = result.AsTensor<float>();
                    var outputDetections = ProcessOutputTensor(outputTensor);
                    if (outputDetections.Count > 0)
                    {
                        detections.AddRange(outputDetections);
                    }
                }

                // Se non ci sono rilevamenti validi, restituisci null
                if (detections.Count == 0)
                {
                    return null;
                }

                // Trova il rilevamento con la confidenza più alta
                var bestDetection = detections.OrderByDescending(d => d.Confidence).First();

                // Calcola le coordinate nel formato originale
                // La chiave qui è considerare il ridimensionamento e il padding che viene applicato
                // quando l'immagine viene preparata per il modello
                var coords = TransformCoordinatesBack(
                    bestDetection.X, bestDetection.Y,
                    originalWidth, originalHeight,
                    scaleInfo);

                return coords;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Errore durante il rilevamento: {ex.Message}");
                Console.WriteLine(ex.StackTrace);
                return null;
            }
        }

        /// <summary>
        /// Prepara l'immagine per l'inferenza.
        /// </summary>
        private DenseTensor<float> PrepareImage(string imagePath, out ScaleInfo scaleInfo)
        {
            using var image = Image.Load<Rgb24>(imagePath);

            // Calcolo fattore di scala per mantenere il rapporto d'aspetto
            float originalWidth = image.Width;
            float originalHeight = image.Height;

            // Calcola il rapporto di scala
            float ratio = Math.Min(_inputWidth / originalWidth, _inputHeight / originalHeight);

            // Calcola nuove dimensioni
            int newWidth = (int)(originalWidth * ratio);
            int newHeight = (int)(originalHeight * ratio);

            // Calcola padding
            int padX = (_inputWidth - newWidth) / 2;
            int padY = (_inputHeight - newHeight) / 2;

            // Resizing con padding - approccio alternativo
            var resizedImage = new Image<Rgb24>(_inputWidth, _inputHeight);

            // Invece di usare Fill, possiamo riempire l'immagine manualmente con nero
            resizedImage.ProcessPixelRows(accessor =>
            {
                for (int y = 0; y < accessor.Height; y++)
                {
                    Span<Rgb24> pixelSpan = accessor.GetRowSpan(y);
                    for (int x = 0; x < accessor.Width; x++)
                    {
                        pixelSpan[x] = new Rgb24(0, 0, 0); // Nero
                    }
                }
            });

            // Ridimensiona l'immagine originale
            image.Mutate(x => x.Resize(newWidth, newHeight));

            // Copia l'immagine ridimensionata nella nuova immagine con padding
            resizedImage.Mutate(ctx => ctx.DrawImage(image, new Point(padX, padY), 1.0f));

            // Converti in tensore
            var tensor = new DenseTensor<float>(new[] { 1, 3, _inputHeight, _inputWidth });

            // Normalizza i pixel
            resizedImage.ProcessPixelRows(accessor =>
            {
                for (int y = 0; y < accessor.Height; y++)
                {
                    Span<Rgb24> pixelSpan = accessor.GetRowSpan(y);
                    for (int x = 0; x < accessor.Width; x++)
                    {
                        tensor[0, 0, y, x] = pixelSpan[x].R / 255.0f; // R
                        tensor[0, 1, y, x] = pixelSpan[x].G / 255.0f; // G
                        tensor[0, 2, y, x] = pixelSpan[x].B / 255.0f; // B
                    }
                }
            });

            // Salva le informazioni di scala per il post-processing
            scaleInfo = new ScaleInfo
            {
                Ratio = ratio,
                PaddingX = padX,
                PaddingY = padY,
                ResizedWidth = newWidth,
                ResizedHeight = newHeight
            };

            return tensor;
        }

        /// <summary>
        /// Informazioni di scala per il post-processing.
        /// </summary>
        private struct ScaleInfo
        {
            public float Ratio;      // Rapporto di scala
            public int PaddingX;     // Padding orizzontale
            public int PaddingY;     // Padding verticale
            public int ResizedWidth;  // Larghezza dopo il ridimensionamento
            public int ResizedHeight; // Altezza dopo il ridimensionamento
        }

        /// <summary>
        /// Trasforma le coordinate dal formato del modello a quello dell'immagine originale.
        /// </summary>
        private (int X, int Y) TransformCoordinatesBack(float x, float y,
            int originalWidth, int originalHeight, ScaleInfo scaleInfo)
        {
            // Rimuovi il padding
            float xNoPad = x - scaleInfo.PaddingX;
            float yNoPad = y - scaleInfo.PaddingY;

            // Se le coordinate sono al di fuori dell'immagine ridimensionata, limitale
            xNoPad = Math.Max(0, Math.Min(xNoPad, scaleInfo.ResizedWidth));
            yNoPad = Math.Max(0, Math.Min(yNoPad, scaleInfo.ResizedHeight));

            // Riscala alle dimensioni originali
            int origX = (int)(xNoPad / scaleInfo.Ratio);
            int origY = (int)(yNoPad / scaleInfo.Ratio);

            // Limita alle dimensioni originali
            origX = Math.Max(0, Math.Min(origX, originalWidth - 1));
            origY = Math.Max(0, Math.Min(origY, originalHeight - 1));

            return (origX, origY);
        }

        /// <summary>
        /// Elabora il tensore di output del modello YOLO ONNX.
        /// </summary>
        private List<Detection> ProcessOutputTensor(Tensor<float> output)
        {
            var detections = new List<Detection>();
            var dimensions = output.Dimensions;

            try
            {
                // Formato [1, 5, 8400] dove le 5 colonne sono [x, y, w, h, confidence]
                if (dimensions.Length == 3 && dimensions[1] == 5)
                {
                    int numDetections = dimensions[2];

                    for (int i = 0; i < numDetections; i++)
                    {
                        float confidence = output[0, 4, i];

                        if (confidence < _confidenceThreshold)
                            continue;

                        // In questo formato, abbiamo solo un tipo di oggetto (mulo)
                        int classId = 0;

                        // Estrai coordinate (x, y) del centro
                        float x = output[0, 0, i];
                        float y = output[0, 1, i];
                        float w = output[0, 2, i];
                        float h = output[0, 3, i];

                        detections.Add(new Detection
                        {
                            ClassId = classId,
                            Confidence = confidence,
                            X = x,
                            Y = y,
                            Width = w,
                            Height = h
                        });
                    }
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Errore nell'elaborazione del tensore: {ex.Message}");
            }

            return detections;
        }

        /// <summary>
        /// Struct per rappresentare un rilevamento.
        /// </summary>
        private struct Detection
        {
            public int ClassId { get; set; }
            public float Confidence { get; set; }
            public float X { get; set; }
            public float Y { get; set; }
            public float Width { get; set; }
            public float Height { get; set; }
        }

        /// <summary>
        /// Rilascia le risorse utilizzate.
        /// </summary>
        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        protected virtual void Dispose(bool disposing)
        {
            if (!_disposed)
            {
                if (disposing)
                {
                    _session?.Dispose();
                }

                _disposed = true;
            }
        }

        ~YoloDetector()
        {
            Dispose(false);
        }
    }
}