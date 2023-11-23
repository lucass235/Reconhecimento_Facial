import cv2
from simple_facerec import SimpleFacerec

# codificar os rosto da pasta
sfr = SimpleFacerec()
sfr.load_encoding_images("../images/faces")

# carregar camera
cap = cv2.VideoCapture(0)


while True:
    ret, frame = cap.read()
    
    # Detectar rostos
    face_locations, face_names = sfr.detect_known_faces(frame)
    for face_loc, name in zip(face_locations, face_names):
        # cordenadas do rosto
        top, left, bottom, right = face_loc[0], face_loc[1], face_loc[2], face_loc[3]
        
        label = "Acesso Liberado"
        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 1
        font_thickness = 2
        text_size, _ = cv2.getTextSize(label, font, font_scale, font_thickness)
        bottom_left = (int((frame.shape[1] - text_size[0]) / 2), frame.shape[0] - 30)
        
        if (name == "Lucas"):
            label = label + " " + name
            cv2.putText(frame, label, bottom_left, font, font_scale, (0, 255, 0), font_thickness)
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        else:
            label = "Acesso Negado" + " " + (name if name != "Unknown" else "Desconhecido")
            cv2.putText(frame, label, bottom_left, font, font_scale, (0, 0, 255), font_thickness)
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            

    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    # tecla S para o loop
    if key == ord('p'):
        break
    
cap.release()
cv2.destroyAllWindows()    