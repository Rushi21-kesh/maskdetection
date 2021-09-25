
from tensorflow.keras.models import load_model
from flask import Flask, render_template, Response
import cv2
from videostream import detect_and_predict_mask
from tensorflow.keras.models import load_model


import logging
  
#Create and configure logger
logging.basicConfig(filename="logs.log",
                    format='%(asctime)s %(message)s',
                    filemode='w')
  
#Creating an object
logger=logging.getLogger()
  
#Setting the threshold of logger to DEBUG
logger.setLevel(logging.DEBUG)

app = Flask(__name__)

prototxtPath = r"face_detector\deploy.prototxt"
weightsPath = r"face_detector\res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# load the face mask detector model from disk
maskNet = load_model("mask_detector.model")

# initialize the video stream
camera = cv2.VideoCapture(0)


def gen_frames():
# loop over the frames from the video stream
    while True:
        success, frame = camera.read()
        logger.info('INFO : Reading Camera')
         # read the camera frame
        if not success:
            logger.info('ERROR : Failed to read')
            break
        else:
             
            
        # grab the frame from the threaded video stream and resize it
        # to have a maximum width of 400 pixels
	        
        # detect faces in the frame and determine if they are wearing a
        # face mask or not
            (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

            # loop over the detected face locations and their corresponding
            # locations
            for (box, pred) in zip(locs, preds):
                # unpack the bounding box and predictions
                (startX, startY, endX, endY) = box
                (mask, withoutMask) = pred

                # determine the class label and color we'll use to draw
                # the bounding box and text
                label = "Mask" if mask > withoutMask else "No Mask"
                color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

                # include the probability in the label
                label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

                # display the label and bounding box rectangle on the output
                # frame
                cv2.putText(frame, label, (startX, startY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
                


            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            

            # show the output frame
        # do a bit of cleanup
       






@app.route('/')
def index():
    
    return render_template('index.html')
    

@app.route('/video_feed')
def video_feed():
    
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
   


if __name__ == '__main__':
    app.run()






