import logging
import os
import time
from datetime import datetime

import cv2
import numpy as np


class TrafficLightDetector:
    def __init__(self, log_file='data/log/traffic_light_log_1.txt'):
        # Konfigurasi logger
        logging.basicConfig(filename=log_file, 
                            level=logging.INFO, 
                            format='%(asctime)s - %(message)s')
        
        # Rentang warna HSV untuk lampu lalu lintas
        self.color_ranges = {
            'red': [
                (0, 100, 150), (10, 255, 255),     # Rentang merah 1
                (160, 190, 150), (180, 255, 255)   # Rentang merah 2
            ],
            'yellow': [(15, 100, 100), (30, 255, 255)],
            'green': [(35, 35, 140), (90, 255, 255)]
        }
        
        # Parameter untuk penanganan kondisi low-light
        self.clahe = cv2.createCLAHE(clipLimit=0.5, tileGridSize=(8, 8))
        
        # Inisialisasi ROI
        self.roi_points = []
        self.drawing = False
        self.rect_start = None
        self.rect_end = None

    def preprocess_frame(self, frame):
        """Preprocessing untuk menangani kondisi pencahayaan rendah"""
        # Konversi ke ruang warna LAB untuk perbaikan kontras
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Perbaikan kontras menggunakan CLAHE
        l2 = self.clahe.apply(l)
        
        # Gabungkan kembali saluran warna
        limg = cv2.merge((l2, a, b))
        frame = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        
        return frame

    def detect_traffic_lights(self, frame, mask):
        """Deteksi lampu lalu lintas menggunakan HSV hanya pada area ROI yang dipilih"""
        # Preprocessing frame
        frame = self.preprocess_frame(frame)
        
        # Konversi ke HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Hasil deteksi
        detections = []
        
        for color, ranges in self.color_ranges.items():
            # Membuat mask untuk setiap rentang warna
            color_mask = cv2.inRange(hsv, ranges[0], ranges[1])
            if len(ranges) > 2:
                mask2 = cv2.inRange(hsv, ranges[2], ranges[3])
                color_mask = cv2.bitwise_or(color_mask, mask2)
            
            # Gunakan mask ROI untuk membatasi deteksi pada area ROI
            masked_color = cv2.bitwise_and(color_mask, color_mask, mask=mask)
            
            # Temukan kontur pada area ROI
            contours, _ = cv2.findContours(masked_color, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter dan gambar bounding box
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 50:  # Filter noise
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    if color == 'red':
                        rectangle_color = (0, 0, 255)  
                        text_color = (0, 0, 255)  
                    elif color == 'yellow':
                        rectangle_color = (0, 255, 255)  
                        text_color = (0, 255, 255)  
                    elif color == 'green':
                        rectangle_color = (0, 255, 0)  
                        text_color = (0, 255, 0)
                    
                    cv2.rectangle(frame, (x, y), (x+w, y+h), rectangle_color, 2)
                    cv2.putText(frame, color.upper(), (x, y-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, text_color, 2)
                    
                    # Log deteksi
                    logging.info(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Detected {color.upper()} light at position {x},{y}")
                    
                    detections.append({
                        'color': color,
                        'position': (x, y),
                        'size': (w, h)
                    })
        
        return frame, detections

    def show_fps(self, frame, prev_time):
        """Menampilkan FPS pada frame"""
        # Menghitung waktu saat ini
        curr_time = time.time()
        
        # Menghitung FPS
        fps = 1 / (curr_time - prev_time)
        
        # Menampilkan FPS di atas frame
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, f'FPS: {fps:.2f}', (10, 30), font, 1, (0, 0, 0), 2, cv2.LINE_AA)
        
        return frame, curr_time

    def select_roi(self, frame):
        """Fungsi untuk memilih ROI secara interaktif dengan drag mouse"""
        cv2.imshow("Select ROI", frame)
        self.roi_points = []
        self.drawing = False
        self.rect_start = None
        self.rect_end = None

        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                # Mulai menggambar ROI
                self.drawing = True
                self.rect_start = (x, y)
            
            elif event == cv2.EVENT_MOUSEMOVE:
                # Perbarui rectangle saat mouse digerakkan
                if self.drawing:
                    self.rect_end = (x, y)
                    frame_copy = frame.copy()
                    cv2.rectangle(frame_copy, self.rect_start, self.rect_end, (0, 255, 0), 2)
                    cv2.imshow("Select ROI", frame_copy)
            
            elif event == cv2.EVENT_LBUTTONUP:
                # Selesaikan menggambar ROI
                self.drawing = False
                self.rect_end = (x, y)
                self.roi_points = [self.rect_start, (self.rect_start[0], self.rect_end[1]), self.rect_end, (self.rect_end[0], self.rect_start[1])]
                cv2.rectangle(frame, self.rect_start, self.rect_end, (0, 255, 0), 2)
                cv2.imshow("Select ROI", frame)

        # Menambahkan callback mouse untuk menangani interaksi
        cv2.setMouseCallback("Select ROI", mouse_callback)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Pastikan 4 titik sudah terpilih
        if len(self.roi_points) == 4:
            return True
        else:
            return False

    def process_video(self, input_video_path, output_video_path):
        """Proses seluruh video"""
        # Buka video input
        cap = cv2.VideoCapture(input_video_path)
        
        # Dapatkan properti video
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Siapkan video output
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        
        # Proses setiap frame
        ret, frame = cap.read()
        if not ret:
            print("Gagal membaca frame.")
            return
        
        # Resize frame menjadi setengah ukuran asli
        frame = cv2.resize(frame, (width // 1, height // 1))

        # Menampilkan UI untuk memilih ROI interaktif
        if not self.select_roi(frame):
            print("ROI tidak dipilih.")
            return

        # Menghitung ROI dari 4 titik
        roi_polygon = np.array(self.roi_points, np.int32)
        roi_polygon = roi_polygon.reshape((-1, 1, 2))

        # Membuat mask berdasarkan ROI
        roi_mask = np.zeros_like(frame, dtype=np.uint8)
        cv2.fillPoly(roi_mask, [roi_polygon], (255, 255, 255))
        roi_mask = cv2.cvtColor(roi_mask, cv2.COLOR_BGR2GRAY)  # Convert ke grayscale mask

        # Waktu awal untuk FPS
        prev_time = time.time()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Resize frame menjadi setengah ukuran asli
            frame = cv2.resize(frame, (width // 1, height // 1))

            # Deteksi lampu lalu lintas pada frame dengan mask ROI
            processed_frame, _ = self.detect_traffic_lights(frame, roi_mask)

            # Tampilkan FPS di frame
            processed_frame, prev_time = self.show_fps(processed_frame, prev_time)

            # Gambar ROI pada frame
            cv2.polylines(processed_frame, [roi_polygon], isClosed=True, color=(0, 255, 255), thickness=2)

            # Tulis frame ke video output
            out.write(processed_frame)

            # Tampilkan frame
            cv2.imshow('Traffic Light Detection', processed_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
        # Tutup semua
        cap.release()
        out.release()
        cv2.destroyAllWindows()


# Contoh penggunaan
if __name__ == '__main__':
    detector = TrafficLightDetector()
    detector.process_video('data/videos/input_video1.mp4', 'data/videos/output_video1.mp4')
