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
        
        cv2.putText(frame, name,(left - 220, top - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    # tecla S para o loop
    if key == ord('p'):
        break
    
cap.release()
cv2.destroyAllWindows()    