import cv2
import time
cam = cv2.VideoCapture(0)
detector=cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

Id=input('Enter Your Id:')
sampleNum=0
while(True):
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        
        #incrementing sample number 
        sampleNum=sampleNum+1
        #saving the captured face in the dataset folder
        cv2.imwrite("camera/user"+Id +'.'+ str(sampleNum) + ".jpg", gray[y:y+h,x:x+w])
        time.sleep(0.1)
        cv2.imshow('frame',img)
    #wait for 100 miliseconds 
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break
    # break if the sample number is morethan 20
    elif sampleNum>100:
        break
cam.release()
cv2.destroyAllWindows()