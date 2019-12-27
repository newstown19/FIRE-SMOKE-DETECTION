from tensorflow.keras.models import load_model
import config
import cv2
from tensorflow.keras.preprocessing import image
import numpy as np
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical
opt = SGD(lr=config.INIT_LR, momentum=0.9,
	decay=config.INIT_LR / config.NUM_EPOCHS)

model  = load_model('firdetection.h5')
model.compile(loss="binary_crossentropy", optimizer=opt,
	metrics=["accuracy"])
cap = cv2.VideoCapture(0)

font = cv2.FONT_HERSHEY_SIMPLEX 
  
 
org = (50, 50) 

fontScale = 1

color = (255, 0, 0) 
  
thickness = 2

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    imagesd = cv2.resize(frame ,(128,128))
    #images = images.astype("uint8")
    images = np.asarray(imagesd , 'float32')
    imagess = np.expand_dims(images , axis = 0)
    pred = model.predict_classes(imagess)
    for i in list(pred):
        if i == 0:
            text = 'FIRE'
        elif i == 1:
            text = 'Neutral'
        else:
            text = 'Smoke'
    
   
    image = cv2.putText(images, text, org, font,  
                   fontScale, color, thickness, cv2.LINE_AA) 
 
    cv2.imshow('frame',image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
