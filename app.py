from flask import Flask, render_template
#from . import collect
import cv2
import numpy as np
from keras.models import model_from_json
import operator
import cv2
import sys, os

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/collect-data')
def collect_data():
    if not os.path.exists("data"):
        os.makedirs("data")
        os.makedirs("data/train")
        os.makedirs("data/test")
        os.makedirs("data/train/0")
        os.makedirs("data/train/1")
        os.makedirs("data/train/2")
        os.makedirs("data/train/3")
        os.makedirs("data/train/4")
        os.makedirs("data/train/5")
        os.makedirs("data/test/0")
        os.makedirs("data/test/1")
        os.makedirs("data/test/2")
        os.makedirs("data/test/3")
        os.makedirs("data/test/4")
        os.makedirs("data/test/5")
    mode = 'train'
    directory = 'data/'+mode+'/'

    cap = cv2.VideoCapture(0)
    while True:
        _, frame = cap.read()
        # Simulating mirror image
        frame = cv2.flip(frame, 1)
    
        # Getting count of existing images
        count = {'Help': len(os.listdir(directory+"/0")),
                'Tea': len(os.listdir(directory+"/1")),
                'Water': len(os.listdir(directory+"/2")),
                'Food': len(os.listdir(directory+"/3")),
                'Doctor': len(os.listdir(directory+"/4")),
                'Thank you': len(os.listdir(directory+"/5"))}
        
        # Printing the count in each set to the screen
        cv2.putText(frame, "MODE : "+mode, (10, 50), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
        cv2.putText(frame, "IMAGE COUNT", (10, 100), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
        cv2.putText(frame, "Help : "+str(count['Help']), (10, 120), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
        cv2.putText(frame, "Tea : "+str(count['Tea']), (10, 140), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
        cv2.putText(frame, "Water : "+str(count['Water']), (10, 160), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
        cv2.putText(frame, "Food : "+str(count['Food']), (10, 180), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
        cv2.putText(frame, "Doctor : "+str(count['Doctor']), (10, 200), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
        cv2.putText(frame, "Thank you : "+str(count['Thank you']), (10, 220), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    
        # Coordinates of the ROI
        x1 = int(0.5*frame.shape[1])
        y1 = 10
        x2 = frame.shape[1]-10
        y2 = int(0.5*frame.shape[1])
        # Drawing the ROI
        # The increment/decrement by 1 is to compensate for the bounding box
        cv2.rectangle(frame, (x1-1, y1-1), (x2+1, y2+1), (255,0,0) ,1)
        # Extracting the ROI
        roi = frame[y1:y2, x1:x2]
        roi = cv2.resize(roi, (64, 64)) 
 
        cv2.imshow("Frame", frame)
    
        #_, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
        #kernel = np.ones((1, 1), np.uint8)
        #img = cv2.dilate(mask, kernel, iterations=1)
        #img = cv2.erode(mask, kernel, iterations=1)
        # do the processing after capturing the image!
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, roi = cv2.threshold(roi, 120, 255, cv2.THRESH_BINARY)
        cv2.imshow("ROI", roi)

        interrupt = cv2.waitKey(10)
        if interrupt & 0xFF == 27: # esc key
            break
        if interrupt & 0xFF == ord('0'):
            cv2.imwrite(directory+'0/'+str(count['HElp'])+'.jpg', roi)
        if interrupt & 0xFF == ord('1'):
            cv2.imwrite(directory+'1/'+str(count['Tea'])+'.jpg', roi)
        if interrupt & 0xFF == ord('2'):
            cv2.imwrite(directory+'2/'+str(count['Water'])+'.jpg', roi)
        if interrupt & 0xFF == ord('3'):
            cv2.imwrite(directory+'3/'+str(count['Food'])+'.jpg', roi)
        if interrupt & 0xFF == ord('4'):
            cv2.imwrite(directory+'4/'+str(count['Doctor'])+'.jpg', roi)
        if interrupt & 0xFF == ord('5'):
            cv2.imwrite(directory+'5/'+str(count['Thank you'])+'.jpg', roi)
    cap.release()
    cv2.destroyAllWindows()
    # Add your Python code for collect data here
    return cv2.VideoCapture(0)

@app.route('/predict')
def predict():
    json_file = open("model-bw.json", "r")
    model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(model_json)

    loaded_model.load_weights("model-bw.h5")
    print("Loaded model from disk")

    cap = cv2.VideoCapture(0)

    # Category dictionary
    categories = {0: 'Help', 1: 'Tea', 2: 'Water', 3: 'food', 4: 'Doctor', 5: 'Thank you'}

    while True:
        _, frame = cap.read()
        # Simulating mirror image
        frame = cv2.flip(frame, 1)
    
        # Got this from collect-data.py
        # Coordinates of the ROI
        x1 = int(0.5*frame.shape[1])
        y1 = 10
        x2 = frame.shape[1]-10
        y2 = int(0.5*frame.shape[1])
        # Drawing the ROI
        # The increment/decrement by 1 is to compensate for the bounding box
        cv2.rectangle(frame, (x1-1, y1-1), (x2+1, y2+1), (255,0,0) ,1)
        # Extracting the ROI
        roi = frame[y1:y2, x1:x2]
    
        # Resizing the ROI so it can be fed to the model for prediction
        roi = cv2.resize(roi, (64, 64)) 
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, test_image = cv2.threshold(roi, 120, 255, cv2.THRESH_BINARY)
        cv2.imshow("test", test_image)
        # Batch of 1
        result = loaded_model.predict(test_image.reshape(1, 64, 64, 1))
        prediction = {'Help': result[0][0], 
                    'Tea': result[0][1], 
                    'Water': result[0][2],
                    'Food': result[0][3],
                    'Doctor': result[0][4],
                    'Thank you': result[0][5]}
        # Sorting based on top prediction
        prediction = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)
    
        # Displaying the predictions
        cv2.putText(frame, prediction[0][0], (10, 120), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)    
        cv2.imshow("Frame", frame)
    
        interrupt = cv2.waitKey(10)
        if interrupt & 0xFF == 27: # esc key
            break
        
 
    cap.release()
    cv2.destroyAllWindows()

    # Add your Python code for prediction here
    return cv2.imshow("Frame", frame)
    

if __name__ == '__main__':
    app.run()
