import cv2
from simple_facerec import SimpleFacerec

# Inicializar o reconhecimento facial
sfr = SimpleFacerec()

# Carregar as imagens codificadas dos rostos da pasta
sfr.load_encoding_images("./images/faces")

# Inicializar a captura de vídeo da câmera (0 para câmera padrão)
cap = cv2.VideoCapture(0)

while True:
    # Capturar um frame da câmera
    ret, frame = cap.read()
    
    # Detectar rostos no frame
    face_locations, face_names = sfr.detect_known_faces(frame)
    
    # Iterar sobre cada rosto detectado
    for face_loc, name in zip(face_locations, face_names):
        # Extrair coordenadas do retângulo que delimita o rosto
        top, left, bottom, right = face_loc[0], face_loc[1], face_loc[2], face_loc[3]
        
        # Configurar texto e estilo da fonte para exibição
        label = "Acesso Liberado,"
        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 1
        font_thickness = 2
        
        # Calcular posição do texto na parte inferior central do frame
        text_size, _ = cv2.getTextSize(label, font, font_scale, font_thickness)
        bottom_left = (int((frame.shape[1] - text_size[0]) / 3.5), frame.shape[0] - 30)
        
        # Verificar se o rosto pertence à pessoa autorizada (Lucas)
        if name != "Desconhecido":
            label = label + " " + name
            cv2.putText(frame, label, bottom_left, font, font_scale, (0, 255, 0), font_thickness)
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        else:
            # Configurar mensagem de acesso negado para rostos desconhecidos ou não autorizados
            label = "Acesso Negado," + " " + name
            cv2.putText(frame, label, bottom_left, font, font_scale, (0, 0, 255), font_thickness)
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

    # Exibir o frame com as informações de reconhecimento facial
    cv2.imshow("Frame", frame)

    # Aguardar por uma tecla (tecla 'p' para sair do loop)
    key = cv2.waitKey(1)
    if key == ord('p'):
        break

# Liberar os recursos da câmera e fechar as janelas abertas
cap.release()
cv2.destroyAllWindows()
