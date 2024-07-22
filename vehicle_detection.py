import cv2 # module or lib
import imutils # module or lib

cascade_src = 'face detection/cars.xml' #import cars cascade
car_cascade = cv2.CascadeClassifier(cascade_src) # read harcascade file

cam = cv2.VideoCapture(0)  # intilizing camera 

while True:
    detected = 0 # starting count with zero
    ret, img = cam.read()  #read frame from camera
    if not ret:
        break
    
    img = imutils.resize(img, width=300) #resize to 300
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #color to grayscale image
    cars = car_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20)) #Coordinate of each & every vechical in the frame
    
    for (x, y, w, h) in cars:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2) # draw the rectangle of the image, color & thickness of the line
    
    cv2.imshow("Frame", img)
    
    detected = len(cars) #length of the car trageted
    n = detected
    print("------------------------------------------------")
    print("North: %d" %(n))
    if n == 0:
        print("No car has been detected")
    if n > 1:
        print("Ford Mustang")
    if n > 2:
        print("Ford Mustang & Dodge")
    if n > 3:
        print("Ford Mustang , Dodge & Corvette")
    if n > 4:
        print("Ford Mustang , Dodge , Corvette & Jaguar E-Type")
    if n > 5:
        print("Lot of cars I can't regonized type of car")
    
    if cv2.waitKey(1) == 27:  # 27 is nothing its esckey 
        break

cam.release()
cv2.destroyAllWindows()
