import numpy as np
import cv2
 
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') # haarcascade_frontalface_default haarcascade_profileface
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')#haarcascade_eye haarcascade_fullbody
    

file = '1.jpg'
img = cv2.imread(file) 
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


faces = face_cascade.detectMultiScale(gray, 1.3, 5)
for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_gray)
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

cv2.namedWindow("Fun", cv2.WINDOW_NORMAL)
cv2.imshow('Fun',img)
cv2.imwrite('./Face'+file, img)
cv2.waitKey(0)
cv2.destroyAllWindows()