import time
import json
import base64
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from ultralytics import YOLO
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

# ===== Serial =====
import serial
import serial.tools.list_ports

app = FastAPI()

# =========================================================
# إعدادات عامة
# =========================================================
ESP_CONF_THRESHOLD = 0.75  # الحد الافتراضي لقبول Enter

# ====== YOLO ======
MODEL_PATH = r"C:\Users\Raidan\Desktop\smart_cashier_local2\model\best.pt"
yolo_model = YOLO(MODEL_PATH)

# =========================================================
# المسارات
# =========================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
FRONTEND_DIR = PROJECT_ROOT / "frontend"
IMAGES_DIR = PROJECT_ROOT / "product_images"
IMAGES_DIR.mkdir(exist_ok=True)

EMB_DIR = PROJECT_ROOT / "embedding"
GALLERY_PATH = EMB_DIR / "gallery.json"

PRODUCTS_FILE = PROJECT_ROOT / "products.json"

# =========================================================
# قاعدة المنتجات + الأسعار
# =========================================================
PRICE_MAP = {
    "Eastroc": 300,
    "Fawar": 500,
    "Noodles": 150,
    "Water": 100,
}

PRODUCTS_DB = []
if PRODUCTS_FILE.exists():
    with open(PRODUCTS_FILE, "r", encoding="utf-8") as f:
        PRODUCTS_DB = json.load(f)

for p in PRODUCTS_DB:
    if "name" in p and "price" in p:
        PRICE_MAP[p["name"]] = p["price"]

# =========================================================
# ربط الواجهة
# =========================================================
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="static")

@app.get("/", response_class=HTMLResponse)
def home():
    index_path = FRONTEND_DIR / "index.html"
    return HTMLResponse(index_path.read_text(encoding="utf-8"))

# =========================================================
# Embedding Model
# =========================================================
def load_embedder():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    weights = MobileNet_V3_Small_Weights.DEFAULT
    model = mobilenet_v3_small(weights=weights)
    model.classifier = torch.nn.Identity()
    model.eval().to(device)
    preprocess = weights.transforms()
    return model, preprocess, device

embed_model, embed_preprocess, EMB_DEVICE = load_embedder()

def load_gallery():
    if not GALLERY_PATH.exists():
        return {}
    with open(GALLERY_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)
    gallery = {}
    for k, v in raw.items():
        gallery[k] = np.array(v["embedding"], dtype=np.float32)
    return gallery

GALLERY = load_gallery()

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / ((np.linalg.norm(a) + 1e-12) * (np.linalg.norm(b) + 1e-12)))

@torch.no_grad()
def crop_to_embedding(bgr_crop: np.ndarray) -> np.ndarray:
    if bgr_crop is None or bgr_crop.size == 0:
        return None
    rgb = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)
    x = embed_preprocess(img).unsqueeze(0).to(EMB_DEVICE)
    feat = embed_model(x).squeeze(0).detach().cpu().numpy().astype(np.float32)
    feat = feat / (np.linalg.norm(feat) + 1e-12)
    return feat

def refine_label_with_embedding(bgr_crop: np.ndarray, base_label: str, threshold: float = 0.88):
    if not GALLERY:
        return base_label, None, None

    vec = crop_to_embedding(bgr_crop)
    if vec is None:
        return base_label, None, None

    best_name = None
    best_score = -1.0
    for name, ref_vec in GALLERY.items():
        # اختياري: نفس الفئة العامة
        if base_label and not name.lower().startswith(base_label.lower()):
            continue

        score = cosine_sim(vec, ref_vec)
        if score > best_score:
            best_score = score
            best_name = name

    if best_name is not None and best_score >= threshold:
        return best_name, best_score, vec

    return base_label, (best_score if best_name else None), vec

def _decode_data_url(data_url: str) -> Optional[np.ndarray]:
    try:
        b64 = data_url.split(",", 1)[1] if "," in data_url else data_url
        img_bytes = base64.b64decode(b64)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None

# =========================================================
# Serial to ESP32
# =========================================================
SERIAL_BAUD = 115200
SERIAL_PORT_FIXED = "COM4"
esp_ser = None

def _open_serial(port: str, baud: int):
    return serial.Serial(port, baud, timeout=0.1, write_timeout=0.2)

def connect_esp32():
    """
    يحاول COM4 أولاً، ثم يبحث بأي COM متاح.
    """
    global esp_ser
    if esp_ser and esp_ser.is_open:
        return

    # 1) جرّب COM4
    try:
        esp_ser = _open_serial(SERIAL_PORT_FIXED, SERIAL_BAUD)
        print(f"[ESP32] Connected on {SERIAL_PORT_FIXED} @ {SERIAL_BAUD}")
        return
    except Exception:
        esp_ser = None

    # 2) ابحث عن أي COM
    ports = serial.tools.list_ports.comports()
    for p in ports:
        if "COM" in p.device:
            try:
                esp_ser = _open_serial(p.device, SERIAL_BAUD)
                print(f"[ESP32] Connected on {p.device} @ {SERIAL_BAUD}")
                return
            except Exception:
                esp_ser = None

    print("[ESP32] NOT connected (no available COM)")

def send_to_esp32(line: str):
    global esp_ser
    try:
        if not esp_ser or not esp_ser.is_open:
            connect_esp32()
        if esp_ser and esp_ser.is_open:
            esp_ser.write((line.strip() + "\n").encode("utf-8", errors="ignore"))
    except Exception as e:
        esp_ser = None
        print("[ESP32] write failed:", e)

@app.on_event("startup")
def _startup():
    connect_esp32()

@app.on_event("shutdown")
def _shutdown():
    global esp_ser
    try:
        if esp_ser and esp_ser.is_open:
            esp_ser.close()
            print("[ESP32] Serial closed")
    except Exception:
        pass

# =========================================================
# Detect: يرجّع detections + invoice
# =========================================================
@app.post("/detect")
async def detect(req: Request):
    payload = await req.json()
    frame_b64 = payload.get("frame")
    conf = float(payload.get("conf", 0.25))
    emb_threshold = float(payload.get("emb_threshold", 0.88))

    if not frame_b64:
        return JSONResponse({"error": "No frame"}, status_code=400)

    frame = _decode_data_url(frame_b64)
    if frame is None:
        return JSONResponse({"error": "Failed to decode image"}, status_code=400)

    results = yolo_model.predict(source=frame, conf=conf, verbose=False)
    r = results[0]

    detections = []
    counts = {}

    h, w = frame.shape[:2]

    if r.boxes is not None and len(r.boxes) > 0:
        boxes = r.boxes.xyxy.cpu().numpy().astype(int)
        confs = r.boxes.conf.cpu().numpy().astype(float)
        clss = r.boxes.cls.cpu().numpy().astype(int)

        for (x1, y1, x2, y2), c, cls_id in zip(boxes, confs, clss):
            # clamp
            x1 = max(0, min(int(x1), w - 1))
            y1 = max(0, min(int(y1), h - 1))
            x2 = max(0, min(int(x2), w - 1))
            y2 = max(0, min(int(y2), h - 1))
            if x2 <= x1 or y2 <= y1:
                continue

            base_label = yolo_model.names.get(int(cls_id), str(cls_id))
            crop = frame[y1:y2, x1:x2]

            refined_label, sim_score, _ = refine_label_with_embedding(
                crop, base_label, threshold=emb_threshold
            )

            detections.append({
                "label": refined_label,
                "base_label": base_label,
                "conf": float(c),
                "bbox": [x1, y1, x2, y2],
                "sim": None if sim_score is None else float(sim_score),
            })

            counts[refined_label] = counts.get(refined_label, 0) + 1

    # ===== Invoice from counts =====
    items = []
    total = 0
    for label, count in counts.items():
        unit = PRICE_MAP.get(label, PRICE_MAP.get(label.split("_")[0], 0))
        subtotal = unit * count
        total += subtotal
        items.append({
            "label": label,
            "count": int(count),
            "unit_price": float(unit),
            "subtotal": float(subtotal)
        })

    return {
        "detections": detections,
        "invoice": {"items": items, "total": float(total)}
    }

# =========================================================
# Commit: يُستدعى عند ضغط Enter فقط
# ويرجّع sent + item متوافق مع الواجهة
# =========================================================
@app.post("/commit")
async def commit(req: Request):
    payload = await req.json()
    detections = payload.get("detections", [])
    threshold = float(payload.get("threshold", ESP_CONF_THRESHOLD))

    if not detections:
        send_to_esp32("ERR")
        return {"sent": "ERR", "reason": "no_detections"}

    # أفضل كشف (أعلى ثقة YOLO)
    def _conf(d):
        try:
            return float(d.get("conf", 0.0))
        except Exception:
            return 0.0

    best = max(detections, key=_conf)
    best_conf = float(_conf(best))

    if best_conf >= threshold:
        label = best.get("label", "Unknown")
        # fallback للسعر العام إذا label فيه suffix
        price = PRICE_MAP.get(label, PRICE_MAP.get(str(label).split("_")[0], 0))

        # إرسال للشاشة
        send_to_esp32(f"OK:{label}:{price}")

        # item للواجهة (حتى تضيفه مباشرة)
        item = {
            "label": label,
            "count": 1,
            "unit_price": float(price),
            "subtotal": float(price)
        }
        return {
            "sent": "OK",
            "label": label,
            "price": float(price),
            "conf": float(best_conf),
            "threshold": float(threshold),
            "item": item
        }

    # مرفوض
    send_to_esp32("ERR")
    return {
        "sent": "ERR",
        "conf": float(best_conf),
        "threshold": float(threshold)
    }

# =========================================================
# Products API
# =========================================================
@app.get("/products")
def get_products():
    return PRODUCTS_DB

@app.post("/add_product")
async def add_product(product_data: dict):
    name = product_data.get("name")
    price = product_data.get("price")
    image_base64 = product_data.get("image")

    if not name or price is None:
        return {"error": "الاسم والسعر مطلوبان"}

    image_name = None
    if image_base64:
        image_name = f"{name}_{int(time.time())}.jpg"
        image_data = base64.b64decode(image_base64.split(",")[1])
        image_path = IMAGES_DIR / image_name
        with open(image_path, "wb") as f:
            f.write(image_data)

    product = {"name": name, "price": price, "image": image_name}
    PRODUCTS_DB.append(product)

    with open(PRODUCTS_FILE, "w", encoding="utf-8") as f:
        json.dump(PRODUCTS_DB, f, ensure_ascii=False, indent=4)

    PRICE_MAP[name] = price
    return {"success": True, "product": product}

@app.get("/product_image/{image_name}")
def get_product_image(image_name: str):
    image_path = IMAGES_DIR / image_name
    if image_path.exists():
        return FileResponse(image_path)
    return {"error": "الصورة غير موجودة"}
