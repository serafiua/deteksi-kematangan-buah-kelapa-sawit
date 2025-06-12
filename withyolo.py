import streamlit as st
from PIL import Image
import numpy as np
import cv2
from scipy.spatial.distance import euclidean
import torch

thresholds = {
    "Belum Masak": {
        "R_norm": 0.331002,
        "G_norm": 0.309469,
        "B_norm": 0.333648,
        "H_norm": 0.415827,
        "S_norm": 0.320983,
        "V_norm": 0.384732
    },
    "Masak": {
        "R_norm": 0.508371,
        "G_norm": 0.320757,
        "B_norm": 0.307987,
        "H_norm": 0.286862,
        "S_norm": 0.528840,
        "V_norm": 0.542386
    },
    "Terlalu Masak": {
        "R_norm": 0.615514,
        "G_norm": 0.400608,
        "B_norm": 0.293272,
        "H_norm": 0.254808,
        "S_norm": 0.573051,
        "V_norm": 0.638349
    }
}

def bgr_to_hsv_manual(b, g, r):
    b = b / 255.0
    g = g / 255.0
    r = r / 255.0
    cmax = max(r, g, b)
    cmin = min(r, g, b)
    delta = cmax - cmin

    if delta == 0:
        h = 0
    elif cmax == r:
        h = (60 * ((g - b) / delta))
    elif cmax == g:
        h = (60 * ((b - r) / delta) + 120)
    else:
        h = (60 * ((r - g) / delta) + 240)

    if h < 0:
        h += 360

    s = 0 if cmax == 0 else (delta / cmax)
    v = cmax

    return h, s * 255, v * 255

def extract_features(img):
    h_img, w_img, _ = img.shape
    total_pixels = h_img * w_img

    B = img[:, :, 0].astype(np.float32)
    G = img[:, :, 1].astype(np.float32)
    R = img[:, :, 2].astype(np.float32)

    mean_R = np.sum(R) / (total_pixels * 255)
    mean_G = np.sum(G) / (total_pixels * 255)
    mean_B = np.sum(B) / (total_pixels * 255)

    H_total, S_total, V_total = 0, 0, 0
    for i in range(h_img):
        for j in range(w_img):
            h, s, v = bgr_to_hsv_manual(B[i, j], G[i, j], R[i, j])
            H_total += h
            S_total += s
            V_total += v

    mean_H = (H_total / total_pixels) / 360.0
    mean_S = (S_total / total_pixels) / 255.0
    mean_V = (V_total / total_pixels) / 255.0

    return [mean_R, mean_G, mean_B], [mean_H, mean_S, mean_V]

def klasifikasi_kematangan(rgb, hsv):
    fitur_input = rgb + hsv
    jarak_min = float("inf")
    kelas_terdekat = None

    for kelas, th in thresholds.items():
        fitur_th = [
            th["R_norm"], th["G_norm"], th["B_norm"],
            th["H_norm"], th["S_norm"], th["V_norm"]
        ]
        jarak = euclidean(fitur_input, fitur_th)
        if jarak < jarak_min:
            jarak_min = jarak
            kelas_terdekat = kelas
    return kelas_terdekat

@st.cache_resource(show_spinner=False)
def load_model():
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    model.conf = 0.5  # confidence threshold
    model.iou = 0.15  # iou threshold
    model.classes = [0]  # misal kalau class palm-fruit ada di 0 (sesuaikan)
    return model

model = load_model()

st.set_page_config(page_title="Deteksi Kematangan Kelapa Sawit", layout="centered")
st.title("🌴 Deteksi Kematangan Buah Kelapa Sawit dengan YOLOv5s")

uploaded_file = st.file_uploader("📸 Upload gambar buah kelapa sawit", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = np.array(Image.open(uploaded_file).convert("RGB"))
    st.image(img, caption="Gambar asli", use_column_width=True)

    # Resize ke 416x416 untuk model YOLO
    img_resized = cv2.resize(img, (416, 416))

    # Deteksi dengan model
    results = model(img_resized)

    # Skala bounding box ke ukuran asli
    h_orig, w_orig, _ = img.shape
    h_resized, w_resized = img_resized.shape[:2]
    scale_x = w_orig / w_resized
    scale_y = h_orig / h_resized

    crops = []
    for *box, conf, cls in results.xyxy[0].cpu().numpy():
        x1, y1, x2, y2 = box
        x1 = int(x1 * scale_x)
        x2 = int(x2 * scale_x)
        y1 = int(y1 * scale_y)
        y2 = int(y2 * scale_y)

        crop = img[y1:y2, x1:x2]
        crops.append(crop)

    if not crops:
        st.warning("Buah kelapa sawit tidak terdeteksi.")
    else:
        for i, crop in enumerate(crops):
            st.image(crop, caption=f"Crop {i+1}", use_column_width=True)

            # Extract fitur (pastikan input BGR untuk fungsi kamu)
            mean_rgb, mean_hsv = extract_features(crop[:,:,::-1])  # RGB->BGR
            hasil = klasifikasi_kematangan(mean_rgb, mean_hsv)
            st.success(f"Hasil klasifikasi Crop {i+1}: **{hasil}**")
