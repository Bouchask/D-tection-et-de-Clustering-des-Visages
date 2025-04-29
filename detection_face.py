import streamlit as st
import cv2
import numpy as np
import torch
import os
import json
import time
import hdbscan
import tempfile
from insightface.app import FaceAnalysis
import mediapipe as mp
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image

# ========== CONFIGURATION ========== 
FACE_SIMILARITY_THRESHOLD = 0.6
FRAME_SKIP = 2
SCALE_FACTOR = 0.5
CLUSTER_MIN_SIZE = 2
# =============================

st.title("📸 Application de Détection et de Clustering des Visages")

uploaded_file = st.file_uploader("📂 Choisissez une vidéo contenant des visages à détecter", type=["mp4", "avi", "mov"])

if uploaded_file:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    input_path = tfile.name

    OUTPUT_FOLDER = tempfile.mkdtemp()
    OUTPUT_VIDEO_PATH = os.path.join(OUTPUT_FOLDER, "output_video.mp4")
    LANDMARKS_JSON_PATH = os.path.join(OUTPUT_FOLDER, "landmarks.json")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(max_num_faces=5, refine_landmarks=True, min_detection_confidence=0.5)

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    saved_embeddings = []
    saved_faces = []
    unique_faces = {}  # Nouveau dictionnaire pour les visages uniques
    face_counter = {}  # Compteur pour chaque visage
    frame_count = 0
    face_count = 0
    landmarks_data = {}

    def enhance_frame(frame):
        frame_lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(frame_lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        frame_lab = cv2.merge((l, a, b))
        return cv2.cvtColor(frame_lab, cv2.COLOR_LAB2BGR)

    def find_similar_face(embedding, saved_embeddings, threshold=FACE_SIMILARITY_THRESHOLD):
        if not saved_embeddings:
            return -1  # Aucun visage trouvé
        similarities = cosine_similarity([embedding], saved_embeddings)
        max_sim_index = np.argmax(similarities)
        if similarities[0][max_sim_index] >= threshold:
            return max_sim_index  # Visage similaire trouvé
        return -1  # Aucun visage similaire

    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_video = cv2.VideoWriter(OUTPUT_VIDEO_PATH, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    start_time = time.time()

    st.info("🔍 Traitement de la vidéo... Veuillez patienter un instant")
    progress = st.progress(0)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % FRAME_SKIP == 0:
            frame = enhance_frame(frame)
            small_frame = cv2.resize(frame, (0, 0), fx=SCALE_FACTOR, fy=SCALE_FACTOR)
            rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

            faces = app.get(rgb_frame)

            for face in faces:
                x1, y1, x2, y2 = [int(v / SCALE_FACTOR) for v in face.bbox]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                face_crop = frame[y1:y2, x1:x2]

                if face_crop.size == 0:
                    continue

                embedding = face.embedding

                # Chercher un visage similaire
                matched_index = find_similar_face(embedding, saved_embeddings)

                if matched_index == -1:
                    # Si aucun visage similaire n'est trouvé, ajouter ce visage à la liste
                    saved_embeddings.append(embedding)
                    saved_faces.append(face_crop)

                    # Extraire les points de repère (landmarks)
                    face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                    results = face_mesh.process(face_rgb)
                    landmarks = []
                    if results.multi_face_landmarks:
                        for lm in results.multi_face_landmarks:
                            for point in lm.landmark:
                                landmarks.append((int(point.x * face_crop.shape[1]), int(point.y * face_crop.shape[0])))

                    landmarks_data[f"face_{face_count}"] = landmarks
                    unique_faces[face_count] = face_crop  # Enregistrer chaque visage unique
                    face_counter[face_count] = 1  # Compter la première détection du visage
                    face_count += 1
                    color = (0, 255, 0)  # Premier visage vu => vert
                else:
                    # Si visage similaire trouvé
                    face_counter[matched_index] += 1  # Augmenter le compteur du visage
                    color = (0, 0, 255)  # Visage déjà vu => rouge
                    cv2.putText(frame, f"Visage vu {face_counter[matched_index]} fois", (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

                # Dessiner le rectangle autour du visage
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        out_video.write(frame)
        frame_count += 1
        progress.progress(min(frame_count / total_frames, 1.0))

    cap.release()
    out_video.release()

    with open(LANDMARKS_JSON_PATH, "w") as f:
        json.dump(landmarks_data, f)

    if saved_embeddings:
        embeddings_array = np.stack(saved_embeddings)
        clusterer = hdbscan.HDBSCAN(min_cluster_size=CLUSTER_MIN_SIZE, metric='euclidean')  # Utilisation de 'euclidean'
        labels = clusterer.fit_predict(embeddings_array)

        for i, label in enumerate(labels):
            if label != -1:
                cluster_folder = os.path.join(OUTPUT_FOLDER, f'person_{label}')
                os.makedirs(cluster_folder, exist_ok=True)
                face_path = os.path.join(cluster_folder, f"face_{i}.jpg")
                cv2.imwrite(face_path, saved_faces[i])

        num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    else:
        num_clusters = 0


   
    with open(OUTPUT_VIDEO_PATH, "rb") as f:
        st.download_button("⬇️ Télécharger la vidéo traitée", f, file_name="processed_faces.mp4")

    with open(LANDMARKS_JSON_PATH, "rb") as f:
        st.download_button("⬇️ Télécharger les données des repères", f, file_name="landmarks.json")
