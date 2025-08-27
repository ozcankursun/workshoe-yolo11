 **Özet (TL;DR):** Küçük veriyle (64/18/9) en iyi hız/doğruluk dengesini **YOLOv11n (50 epoch)** sağladı.  

> **Val mAP@50–95:** `0.669` • **Test mAP@50–95:** `0.553` • **~FPS (CPU, 512):** `8–12`


---

## ✨ Kilit Noktalar
- **Seçilen model:** `yolo11n` (nano) — küçük boyut, yüksek hız; doğruluk kaybı yok denecek kadar az.  
- **Kıyas:** `yolo11s` daha yavaş ve bu veri boyutunda daha düşük mAP/recall.  
- **Kaynak veri:** YouTube videosu → kare çıkarma → Roboflow etiketleme (tek sınıf: `work_shoe`).  

---

## 🗂️ Veri Seti
- **Kaynak:** `yt-dlp` ile video indirildi, **OpenCV** ile karelere bölündü.  
- **Ölçek:** Toplam ~3,265 kare → eşit aralıklı **600** seçildi → **~100** görsel etiketlendi.  
- **Sınıflar:** `work_shoe` (1 sınıf)  
- **Split:** **Train/Val/Test = 64 / 18 / 9** (≈ **70/20/10**)  
- **Format:** YOLO (Ultralytics/YOLOv8)

```
data/workshoes/
  images/train   labels/train
  images/val     labels/val
  images/test    labels/test
```

> **Etiketleme kuralı:** Örtüşmelerde (occlusion) yalnızca **görünen kısmı** kutula.

---

## 🧪 Deney Akışı
```mermaid
flowchart LR
    A[Video] --> B[Frame Extraction]
    B --> C[Roboflow Labeling]
    C --> D[Train (YOLOv11n/s)]
    D --> E[Validation Metrics]
    E --> F[Test Metrics]
    D --> G[ONNX Export]
    F --> H[Report + Examples]
```

---

## 📊 Sonuçlar (Validation)
| Model     | Epoch | Precision | Recall  | mAP@50  | mAP@50–95 | Inference (ms/img) | ~FPS |
|-----------|------:|----------:|--------:|--------:|----------:|-------------------:|-----:|
| yolo11n   |   10  | 0.8754    | 0.7917  | 0.9038  | 0.6024    | 119.29             | 8.4  |
| **yolo11n** | **50** | **0.9581**  | **0.9533**| **0.9876**| **0.6695** | **121.29**           | **8.2** |
| yolo11s   |   50  | 1.0000    | 0.8709  | 0.9698  | 0.6303    | 229.01             | 4.4  |

> **Predict hız notu:** ~**86.2 ms/img ≈ 11.6 FPS** @512 (bazı koşullarda daha hızlı).

### 🧪 Test (Seçilen model: **YOLOv11n @ 50e**)
- **Precision:** `0.9974`  

- **Recall:** `0.8889`  

- **mAP@50:** `0.9197`  

- **mAP@50–95:** `0.5530`

**Yorum:** Model **temkinli** (yüksek precision); farklı ışık/açıda bazı ayakkabıları kaçırabiliyor (recall↓). Inference’ta `--conf 0.35` denemesi recall’ı artırıyor (FP artışı izlenmeli).

---

## ⚖️ Trade‑off (Boyut ↔ Doğruluk) — **Karar**
> **Sonuç:** **YOLOv11n** bu veri boyutunda **daha hızlı** ve **daha yüksek** mAP@50–95/recall verdi.  

> **Karar:** Görev kısıtına uygun şekilde **nihai model = YOLOv11n (50e)** ✅

**Gerekçe:** `yolo11s` precision 1.0 olsa da **recall** ve **mAP@50–95** düşüyor; **inference ~2× yavaş** (CPU).

---