import streamlit as st
from ultralytics import YOLO
import cv2
from PIL import Image
import numpy as np

# Configuración de la página
st.set_page_config(page_title="Detector de Coches", layout="centered")
st.title("🚗 Detector de Vehículos")

# Cargamos el modelo YOLOv8 pequeño (rápido y eficiente)
@st.cache_resource
def load_model():
    return YOLO('yolov8n.pt')

model = load_model()

# Filtro: IDs de clases en YOLO para vehículos
# 2: car, 3: motorcycle, 5: bus, 7: truck
vehicle_classes = [2, 3, 5, 7]

img_file = st.camera_input("Apunta a la carretera y haz una foto")

if img_file:
    # Convertir imagen para que YOLO la entienda
    img = Image.open(img_file)
    img_array = np.array(img)
    
    # Realizar detección filtrando solo por vehículos
    results = model(img_array, classes=vehicle_classes)
    
    # Dibujar las bounding boxes
    res_plotted = results[0].plot()
    
    # Mostrar resultado
    st.image(res_plotted, caption="Detección finalizada", use_container_width=True)
    st.success(f"Se han detectado {len(results[0].boxes)} vehículos.")