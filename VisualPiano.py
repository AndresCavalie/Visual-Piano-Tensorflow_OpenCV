import cv2
import numpy as np
import keyboard

import tensorflow as tf
from tensorflow import keras


model = keras.models.load_model("model.h5")
def stackImages(scale,imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver

cap = cv2.VideoCapture(0)
def empty(a):
    pass

cv2.namedWindow("parameters")
cv2.resizeWindow("parameters", 640,300)
cv2.createTrackbar("Lower H","parameters", 75,180 , empty)
cv2.createTrackbar("Lower S","parameters", 41,255 , empty)
cv2.createTrackbar("Lower V","parameters", 0,255 , empty)
cv2.createTrackbar("Upper H","parameters", 87,180 , empty)
cv2.createTrackbar("Upper S","parameters", 255,255 , empty)
cv2.createTrackbar("Upper V","parameters", 255,255 , empty)
#instance=0
imgs = []
noteboxes = []
while cap.isOpened():
    #instance+=1
    hasFrame, img = cap.read()
    #img = cv2.imread("numbers.png")
    
    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    
    lh = cv2.getTrackbarPos("Lower H","parameters")
    ls = cv2.getTrackbarPos("Lower S","parameters")
    lv = cv2.getTrackbarPos("Lower V","parameters")
    uh = cv2.getTrackbarPos("Upper H","parameters")
    us = cv2.getTrackbarPos("Upper S","parameters")
    uv = cv2.getTrackbarPos("Upper V","parameters")

    
    
    
    
    lower_red = np.array([lh,ls,lv])
    upper_red = np.array([uh,us,uv])
    
    
    mask = cv2.inRange(hsv, lower_red, upper_red)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    copy = np.copy(img)
    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        # approx = cv2.approxPolyDP(cnt, 0.15 * peri, True)
        # area=cv2.contourArea(approx)
        
        #for i=0; i<len(approx); i++:
        if peri> 20:
            #cv2.drawContours(img, [cnt], -1 , (255,0,0),3)
            count= 0
            if count<8:
                # print("APPROX: ")
                # print(approx)
                # print("SIDES")
                # print(len(approx))
                count += 1
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(img,(x-10,y-10),(x+w+10,y+h+10),(255,0,0))
            
            
            
            # if len(approx) == 4:
            #     x, y, w, h = cv2.boundingRect(approx)
            #     cv2.putText(img, "piano" , (x+w+10, y+10), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0,255,0), 2)
            # if len(approx) == 2:
            #     x, y, w, h = cv2.boundingRect(approx)
                
            #     cv2.putText(img, "bass" , (x+w+10, y+10), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0,255,0), 2)
            # if len(approx) == 3:
            #     x, y, w, h = cv2.boundingRect(approx)
                
            #     cv2.putText(img, "drum" , (x+w+10, y+10), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0,255,0), 2)
        

                   
    
    
    
    
    
    kernel = np.ones((4,4))
    imgDil = cv2.dilate(mask, kernel,iterations =1)
    #imgStack = stackImages(0.5,([img,mask,imgDil]))
    cv2.putText(img, "Use the HSV color sliders to isolate your numbers," , (20, 20), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255,255,255), 1)
    cv2.putText(img, "Then, press Esc to continue!" , (20,40), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255,255,255), 1)
    cv2.imshow("FILTER",img)

    #cv2.imshow("mask", mask)
    # print(instance)
    if cv2.waitKey(5) & 0xFF == 27:

      
      break 
cap.release()
cv2.destroyWindow("FILTER")
cv2.destroyWindow("parameters")
    # count = 0
    # if keyboard.read_key() == "s":
    #if instance == 500:
i = 0
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    #inverted = cv2.bitwise_not(mask)
    inverted = mask
    if (w+20)*(h+20) > 900:
        if w>h:
            difference = w-h
            dif = int(difference/2)
            pad = 0
            if (dif*2) + h != w:
                pad = abs(w - ((dif*2) + h))
            crop = inverted[y-dif-10:y+h+dif+10+pad, x-10:x+w+10]
            x2 = x-10
            w2 = (x+w+10)-(x-10)
            
        elif h>w:
            difference = h-w
            dif = int(difference/2)
            pad = 0
            if (dif*2) + w != h:
                pad = abs(h - ((dif*2) + w))
            crop = inverted[y-10:y+h+10, x-10-dif:x+w+dif+10+pad]
            x2 = x-10-dif
            w2 = (x+w+dif+10+pad) - (x-10-dif)
        else:
            crop = inverted[y-10:y+h+10, x-10:x+w+10]
            x2 = x-10
            w2 = (x+w+10) - (x-10)
        
        crop = cv2.resize(crop, (28,28))
        
        imgs.append(crop)
        print("should correspond to image"+str(i)+".png")
        noteboxes.append((x2,y,w2,h))
        print((x2,y,w2,h))
    
        # print("imgs array")
        # print(imgs)
        # count += 1
        #crop = inverted[y-10:y+h+10, x-10:x+w+10]
        savecrop = crop
        name = "image"+str(i)+".png"
        reader = crop/255
        # print(reader)
        

        cv2.imwrite(name, crop)
        # print(i)
        i += 1
                
                

  ##  for number in range(len(fields)):
       # number = fields[number][0]
     
        
    

    

shapeSize = len(imgs)

# print(shapeSize)
numbers = np.zeros(shape=(shapeSize,28,28))

imgs = np.array([imgs])
# print("IMGS")
# print(imgs[0])
# print("savecrop")
# print(imgs[0].shape)
# print(imgs[0][0])
imgs = imgs[0]
#print(savecrop)
for count, pic in enumerate(imgs):
    # print("PRE NUMPY")
    # print(pic)
    # print("POST NUMPY")
    pic = np.array([pic])
    # print(pic)
    pic = pic/255
#     print("numpy convert")
#     print(pic)
    numbers[count] = pic


from Prediction import getPreds


notes = getPreds(numbers,model)
print(notes)
print(noteboxes)









# import cv2
import mediapipe as mp
import rtmidi

NOTE_VELOCITY = 127
WINDOW_NAME = "MotionPiano"
KEY_HEIGHT = 0.25


midiout = rtmidi.MidiOut()

assert(midiout.get_ports())
portNumber = 0 if len(midiout.get_ports()) == 1 or 'through' not in str(midiout.get_ports()[0]).lower() else 1
midiout.open_port(portNumber)

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# For static images:
IMAGE_FILES = []
with mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.5) as hands:
  for idx, file in enumerate(IMAGE_FILES):
    # Read an image, flip it around y-axis for correct handedness output (see
    # above).
    image = cv2.flip(cv2.imread(file), 1)
    # Convert the BGR image to RGB before processing.
    results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Print handedness and draw hand landmarks on the image.
    print('Handedness:', results.multi_handedness)
    if not results.multi_hand_landmarks:
      continue
    image_height, image_width, _ = image.shape
    annotated_image = image.copy()
    for hand_landmarks in results.multi_hand_landmarks:
      print('hand_landmarks:', hand_landmarks)
      print(
          f'Index finger tip coordinates: (',
          f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width}, '
          f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height})'
      )
      mp_drawing.draw_landmarks(
          annotated_image,
          hand_landmarks,
          mp_hands.HAND_CONNECTIONS,
          mp_drawing_styles.get_default_hand_landmarks_style(),
          mp_drawing_styles.get_default_hand_connections_style())
    cv2.imwrite(
        '/tmp/annotated_image' + str(idx) + '.png', cv2.flip(annotated_image, 1))
    # Draw hand world landmarks.
    if not results.multi_hand_world_landmarks:
      continue
    for hand_world_landmarks in results.multi_hand_world_landmarks:
      mp_drawing.plot_landmarks(
        hand_world_landmarks, mp_hands.HAND_CONNECTIONS, azimuth=5)

# For webcam input:
cap = cv2.VideoCapture(0)
NOTES = [ 60, 63, 65, 67, 68, 70, 72, 73, 77, 75 ]
notenums = notes
noteBoxes = noteboxes

boxval = [[False,None] for _ in range(len(noteBoxes))]
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

   # for box in noteBoxes:
           # cv2.rectangle(image,(box[0],box[1]),(box[0]+box[2],box[1]+ box[3]),(255,0,0))
            
    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())
        
        #4 8 12 16 20
        lmList = []
        
        for hand in results.multi_hand_landmarks:
          for id, lm in enumerate(hand.landmark):
                  h, w, c = image.shape
                  cx, cy = int(lm.x*w), int(lm.y*h)
                  if id==8:
                #   if id==4 or id==8 or id==12 or id==16 or id==20:
                    lmList.append([id, cx, cy])
            
        #print("INDEX (8?): "+str(lmList[8][0])+"  xpos: " + str(lmList[8][1])+"  ypos: " + str(lmList[8][2]))
       # handx = lmList[8][1]
        #handy = lmList[8][2]
        
        
        
        print(boxval)
        for i in range(len(noteBoxes)):
          print(boxval[i][0])
          print(boxval[i][1])
          for j in range(len(lmList)):
            
            if noteBoxes[i][0] < lmList[j][1] < noteBoxes[i][0]+noteBoxes[i][2] and noteBoxes[i][1] < lmList[j][2] < noteBoxes[i][1]+ noteBoxes[i][3]:
              print('IN RANGE : ')
              print(boxval[i][0])
              print("")
              print("")
              if boxval[i][0]==False:
                midiout.send_message([0x90, NOTES[int(notenums[i])], NOTE_VELOCITY])
                #Hack fix
                print("box set to true")
                boxval[i][0]=True
                print(j)
                boxval[i][1]=str(j)
                print(boxval[i][0])
                print('finger ID')
                print(boxval[i][1])
                
                
              
            elif str(j)==boxval[i][1]: #work on this its fucked up
              print("box set to False")
              boxval[i][0]=False
              print(boxval[i][0])
              
        #THIS IS NOT WORKING AGAIN, probably should just use method
                
                
            
        
          

        
        
        
        
        
        
    # Flip the image horizontally for a selfie-view display.
    cv2.putText(image, "Use either of your index fingers to play!" , (20,40), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255,255,255),1)
    cv2.imshow('MediaPipe Hands', image)
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()










