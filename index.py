import streamlit as st
import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import pandas as pd
import joblib
import json


st.title("Cricket Player Classifier")
st.markdown(""" 
            1. Virat Kohli \n
            2. Mahendra Singh Dhoni \n
            3. Rishabh Pant \n
            4. Rohit Sharma \n
            5. Jaspreet Bumrah \n
            6. Yuzi Chahal\n""")
st.markdown("""## This classifier takes a picture of one of the cricketers mentioned above and returns the name of that cricketer.""")
inp_img = st.file_uploader("Choose file")

face_cascade = cv2.CascadeClassifier("C:/Users/aryan/ML PROJECT/CricketPlayerClassifier/model/openCV/haarCascades/haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("C:/Users/aryan/ML PROJECT/CricketPlayerClassifier/model/openCV/haarCascades/haarcascade_eye.xml")

def get_cropped_image_if_2_eyes(img):
    if (img is not None):
        gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray,1.3,5)
        for (x,y,w,h) in faces:
            roi_gray = gray[y:y+h,x:x+w]
            roi_color = img[y:y+h,x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            if len(eyes)>=2:
                return roi_color
            
import pywt  

def w2d(img, mode='haar', level=1):
    imArray = img
    #Datatype conversions
    #convert to grayscale
    imArray = cv2.cvtColor( imArray,cv2.COLOR_RGB2GRAY )
    #convert to float
    imArray =  np.float32(imArray)   
    imArray /= 255
    # compute coefficients 
    coeffs=pywt.wavedec2(imArray, mode, level=level)

    #Process Coefficients
    coeffs_H=list(coeffs)  
    coeffs_H[0] *= 0  

    # reconstruction
    imArray_H=pywt.waverec2(coeffs_H, mode)
    imArray_H *= 255
    imArray_H =  np.uint8(imArray_H)

    return imArray_H

model = joblib.load("C:/Users/aryan/ML PROJECT/CricketPlayerClassifier/model/saved_model.pkl")
x= []
dict = {'Bumrah ': 0,
 'Chahal ': 1,
 'Mahendra Singh Dhoni ': 2,
 'Rishabh Pant ': 3,
 'Rohit Sharma ': 4,
 'Virat Kohli ': 5}
if inp_img is not None :
    read = Image.open(inp_img)
    img = np.array(read)           
    roi_color = get_cropped_image_if_2_eyes(img)
    if roi_color is None:
        st.write("## unable to process image, try some other image")
    else:
        scalled_raw_img= cv2.resize(roi_color,(32,32))
        img_har = w2d(roi_color,"db1",5)
        scalled_img_har= cv2.resize(img_har,(32,32))
        combined_img = np.vstack((scalled_raw_img.reshape(32*32*3,1),scalled_img_har.reshape(32*32,1)))
        x.append(combined_img)
        X= np.array(x).reshape(len(x),len(x[0])).astype("float")
        answer = model.predict(X)
        for i,j in dict.items():
            if j==answer[0]:
                st.write("## ",i) 

