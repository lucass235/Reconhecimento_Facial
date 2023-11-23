import cv2
import face_recognition

# Aqui carregamos a imagem que queremos comparar
img = cv2.imread("./images/Djalma2.png")
rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_encoding = face_recognition.face_encodings(rgb_img)[0]

# Aqui carregamos a 2 imagem que queremos comparar
img2 = cv2.imread("./images/Djalma.jpg")
rgb_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
img_encoding2 = face_recognition.face_encodings(rgb_img2)[0]

# Aqui comparamos as duas imagens
result = face_recognition.compare_faces([img_encoding], img_encoding2)
print("Resultado: ", result)



cv2.imshow("Img", img)
cv2.imshow("Img2", img2)
cv2.waitKey(0)