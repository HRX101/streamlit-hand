import streamlit as st
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import numpy as np
import cv2
import mediapipe as mp
from streamlit_webrtc import RTCConfiguration, webrtc_streamer,VideoTransformerBase
st.title("hand-detection")
frame_window=st.image([])
run=st.checkbox("instruction")
if run:
    st.write("press start for detection")
p12=""
convnet = input_data(shape=[50,50,1])
convnet = conv_2d(convnet, 32, 5, activation='relu')
# 32 filters and stride=5 so that the filter will move 5 pixel or unit at a time
convnet = max_pool_2d(convnet, 5)
convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)
convnet = conv_2d(convnet, 128, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)
convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet, 0.8)
convnet = fully_connected(convnet, 11, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate = 0.001, loss='categorical_crossentropy')
model = tflearn.DNN(convnet, tensorboard_verbose=1)
model.load("model.tflearn")
mphands=mp.solutions.hands
hands=mphands.Hands()
mpDraw=mp.solutions.drawing_utils
my_label=""
l=""
mphands=mp.solutions.hands
hands=mphands.Hands()
p=0
l=""
pas=False
p1=""
m=[]
mpDraw=mp.solutions.drawing_utils
class VideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        img= frame.to_ndarray(format="bgr24")
       
        img=cv2.flip(img,1)
        gray1=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        results=hands.process(gray1)
       
        if results.multi_hand_landmarks:
            for hand_frame in results.multi_hand_landmarks:
                mpDraw.draw_landmarks(img,hand_frame,mphands.HAND_CONNECTIONS)
                cv2.rectangle(img, (200,40),
                                  (450,350),
                                  (255, 0, 255), 2)
                pr=frame
                roi = img[40:350,200:450]
                roi = cv2.resize(roi, (200, 200))
                lower = np.array([200, 200, 200])
                upper = np.array([255, 255, 255])
                thresh = cv2.inRange(roi, lower, upper)
                detected_face = cv2.resize( thresh,(50,50))
                #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20,20))
                #detected_face= cv2.morphologyEx(detected_face, cv2.MORPH_CLOSE, kernel)
                
                detected_face = np.array(detected_face).reshape((-1,50,50,1))
        
       # label = np.argmax(model.predict(detected_face))
                conf=model.predict(detected_face)[0]
                idx=np.argmax(conf)
            
                confiedence="{:.2f}%".format(conf[idx]*100)
                model_out=model.predict(detected_face)
            #print(model_out)
            
                if np.argmax(model_out) == 0:
                    my_label = 'A'
                    m.append(my_label)
                    st.write(my_label)
                
                elif np.argmax(model_out) == 1:
                    my_label = 'B'
                    m.append(my_label)
                    st.write(my_label)
                elif np.argmax(model_out) == 2:
                    my_label = 'C'
                    m.append(my_label)
                    st.write(my_label)
                elif np.argmax(model_out) == 3:
                    my_label = 'D'
                    m.append(my_label)
                    st.write(my_label)
                elif np.argmax(model_out) == 4:
                    my_label = 'E' 
                    m.append(my_label)
                    st.write(my_label)
                elif np.argmax(model_out) == 5:
                    my_label = 'F' 
                    m.append(my_label)
                    st.write(my_label)
                #os.system('shutdown/s/t 10')   
                elif np.argmax(model_out) == 6:
                    my_label = 'G'
                    m.append(my_label)
                    st.write(my_label)
                elif np.argmax(model_out) == 7:
                    my_label = 'H'
                    m.append(my_label)
                    st.write(my_label)
                #pyautogui.hotkey('win','d')   
                elif np.argmax(model_out) == 8:
                    my_label = 'I'
                    m.append(my_label)
                    
                elif np.argmax(model_out) == 9:
                    my_label = 'J' 
                    m.append(my_label)
                #os.system('shutodown/r/t 10')
                elif np.argmax(model_out) == 10:
                    my_label = 'K'    
                    m.append(my_label) 
                else:
                    my_label = 'unkwnon'
                    m.append(my_label)
            #print(my_label)
                cv2.putText(img,my_label,(100,100),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),1)
                cv2.putText(img,str(p),(120,120),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0))
                cv2.putText(img,str(confiedence),(120,180),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0))
               
        return img

st.write(m)
webrtc_streamer(key="key", video_transformer_factory=VideoTransformer,
            rtc_configuration=RTCConfiguration(
                {"iceServers":[{"urls":["stun:stun.l.google.com:19302"]}]}
            )
)
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header{visibility:hidden;}
            a{
                visibility:hidden;
            }
  
            footer:after{
                visibility:visible;
                content:'Made by HP';
                display:block;
                color:red;
                padding:5px;
                top:3px;
            }
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 
