import cv2

#camera object
cam = cv2.VideoCapture(0)


fileName = input("What is your name?")
dataset_path = "./data"

model = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

offest = 30 #Extra reagion to include


while True:
    success, img = cam.read()
    cropped_face = img
    if not success:
        print("not able to show")
        break
    # cropped_face = img
    greyImg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = model.detectMultiScale(img,1.5,2)
    faces = sorted(faces,key=lambda f:f[2]*f[3])
    if len(faces) > 0:
        f = faces[-1]
        x,y,w,h = f
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2 )
        cropped_face = img[y : y+h, x   :x+w]

    cv2.imshow("Pratham",img)
    cv2.imshow("Cropped face", greyImg  )
    key = cv2.waitKey(1)
    if(key == ord('q')):
        print("Exitings the code")
        break

cam.release()
cv2.destroyAllWindows()