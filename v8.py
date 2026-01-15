import cv2
from ultralytics import YOLO

def webcam_detect_simple(weights='bestofbest.pt'):
    # Modeli yükle
    model = YOLO(weights)
    
    # Webcam'i başlat
    cap = cv2.VideoCapture(0)
    
    print("Webcam başlatıldı. Çıkmak için 'q' tuşuna basın...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # YOLOv8 ile tahmin yap
        results = model(frame, conf=0.25, verbose=False)
        
        # Sonuçları görselleştir
        annotated_frame = results[0].plot()
        
        # Görüntüyü göster
        cv2.imshow('YOLOv8 Detection', annotated_frame)
        
        # Çıkış için 'q' tuşu
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    webcam_detect_simple('bestofbest.pt')  # model dosyanızın adını buraya yazın