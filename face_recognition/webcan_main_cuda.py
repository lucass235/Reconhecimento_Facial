import cv2
from simple_facerec import SimpleFacerec

# Verifique se o OpenCV foi compilado com suporte para GPU
print(cv2.getBuildInformation())

# Codificar os rostos da pasta
sfr = SimpleFacerec()
sfr.load_encoding_images("./images/faces")

# Carregar a câmera
cap = cv2.VideoCapture(0)

# Definir o backend para CUDA se disponível
if cv2.cuda.getCudaEnabledDeviceCount() > 0:
    print("Usando a GPU")
    cv2.dnn.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    cv2.dnn.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
else:
    print("Usando CPU")    

while True:
    ret, frame = cap.read()
    
    # Detectar rostos
    face_locations, face_names = sfr.detect_known_faces(frame)
    for face_loc, name in zip(face_locations, face_names):
        # Coordenadas do rosto
        top, left, bottom, right = face_loc[0], face_loc[1], face_loc[2], face_loc[3]
        
        label = "Acesso Liberado,"
        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 1
        font_thickness = 2
        text_size, _ = cv2.getTextSize(label, font, font_scale, font_thickness)
        bottom_left = (int((frame.shape[1] - text_size[0]) / 2), frame.shape[0] - 30)
        
        if (name != "Desconhecido"):
            label = label + " " + name
            cv2.putText(frame, label, bottom_left, font, font_scale, (0, 255, 0), font_thickness)
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        else:
            label = "Acesso Negado," + " " + name
            cv2.putText(frame, label, bottom_left, font, font_scale, (0, 0, 255), font_thickness)
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            

    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    # Tecla 'p' para sair do loop
    if key == ord('p'):
        break
    
cap.release()
cv2.destroyAllWindows()
