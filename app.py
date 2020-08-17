import streamlit as st 
import cv2
from PIL import Image,ImageEnhance
import numpy as np 
import os
from model import FacialExpressionModel
import time
from bokeh.models.widgets import Div

#importing the cnn model+using the CascadeClassifier to use features at once to check if a window is not a face region
st.set_option('deprecation.showfileUploaderEncoding', False)
face_cascade = cv2.CascadeClassifier('frecog/haarcascade_frontalface_default.xml')
model = FacialExpressionModel("my_model/model.json", "my_model/model_weights.h5")
font = cv2.FONT_HERSHEY_SIMPLEX 

#face exp detecting function
def detect_faces(our_image):
	new_img = np.array(our_image.convert('RGB'))
	img = cv2.cvtColor(new_img,1)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	# Detect faces
	faces = face_cascade.detectMultiScale(gray, 1.1, 4)
	# Draw rectangle around the faces
	for (x, y, w, h) in faces:

			fc = gray[y:y+h, x:x+w]
			roi = cv2.resize(fc, (48, 48))
			pred = model.predict_emotion(roi[np.newaxis, :, :, np.newaxis])
			cv2.putText(img, pred, (x, y), font, 1, (255, 255, 0), 2)
			cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
			return img,faces,pred 

#cartonizing_image function
def cartonize_image(our_image):
	new_img = np.array(our_image.convert('RGB'))
	img = cv2.cvtColor(new_img,1)
	gray = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
	# Edges
	gray = cv2.medianBlur(gray, 5)
	edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
	#Color
	color = cv2.bilateralFilter(img, 9, 300, 300)
	#Cartoon
	cartoon = cv2.bitwise_and(color, color, mask=edges)

	return cartoon

#cannizing_image function
def cannize_image(our_image):
	new_img = np.array(our_image.convert('RGB'))
	img = cv2.cvtColor(new_img,1)
	img = cv2.GaussianBlur(img, (11, 11), 0)
	canny = cv2.Canny(img, 100, 150)
	return canny
#the main function	

def main():
	
	"""Face Expression Detection App"""
	#setting the app title & sidebar

	st.title("Face  Expression  Detection Application")

	st.markdown(":smile: :worried: :fearful: :rage: :hushed:") 
	#st.text("Built with Streamlit and OpenCV")

	activities = ["Home","Detect your Facial expressions" ,"CNN Model Performance","About"]
	choice = st.sidebar.selectbox("Select Activity",activities)
	#using Html and Css for home page
	if choice == 'Home':
		html_temp = """
		<div style=" 
					background-image: linear-gradient(to right, #FBCEB1, #733635);
					width: 700px;
					height:500px;
					border-radius: 25px;
					opacity:0.2;">
		</div><br>
		"""
	
		st.markdown(html_temp, unsafe_allow_html=True)

	if choice == 'CNN Model Performance':
		st.image('model.png', width=850)
	#if choosing the first choice on sidebar , give access to upload the image 
			

	if choice == 'Detect your Facial expressions':
		st.subheader("Face Detection")

		image_file = st.file_uploader("Upload Image",type=['jpg','png','jpeg'])
    
	#if image if uploaded,display the progress bar +the image
		if image_file is not None:
				our_image = Image.open(image_file)
				st.text("Original Image")
				progress = st.progress(0)
				for i in range(100):
					time.sleep(0.01)
					progress.progress(i+1)
				st.image(our_image)
		if image_file is None:
			st.error("No image uploaded yet")

		# Face Detection
		task = ["Faces","Cannize","Cartonize"]
		feature_choice = st.sidebar.selectbox("Find Features",task)
		if st.button("Process"):
			if feature_choice == 'Faces':

				#process bar
				progress = st.progress(0)
				for i in range(100):
					time.sleep(0.05)
					progress.progress(i+1)
				#end of process bar
				
				result_img,result_faces,prediction = detect_faces(our_image)
				if st.image(result_img) :
					st.success("Found {} faces".format(len(result_faces)))
					#recommended youtube videos for the user
					if prediction == 'Happy':
						st.subheader("YeeY!  You are Happy :smile: today , Always Be ! ")
						st.text("Here is your Recommended video to watch:")
						st.video("https://www.youtube.com/watch?v=M1uyH-DzjGE&t=46s")
					elif prediction == 'Angry':
						st.subheader("You seem to be angry :rage: today ,Take it easy! ")
						st.text("Here is your Recommended video to watch:")
						st.video("https://www.youtube.com/watch?v=M1uyH-DzjGE&t=46s")
					elif prediction == 'Disgust':
						st.subheader("You seem to be Disgust :rage: today ,Take it easy! ")
						st.text("Here is your Recommended video to watch:")
						st.video("https://www.youtube.com/watch?v=M1uyH-DzjGE&t=46s")
					elif prediction == 'Fear':
						st.subheader("You seem to be Fearful :fearful: today ,Be couragous! ")
						st.text("Here is your Recommended video to watch:")
						st.video("https://www.youtube.com/watch?v=M1uyH-DzjGE&t=46s")
					elif prediction == 'Neutral':
						st.subheader("You seem to be Neutral :neutral: today ,Happy day ")
						st.text("Here is your Recommended video to watch:")
						st.video("https://www.youtube.com/watch?v=M1uyH-DzjGE&t=46s")
					elif prediction == 'Sad':
						st.subheader("You seem to be Sad :sad: today ,Smile and be happy! ")
						st.text("Here is your Recommended video to watch:")
						st.video("https://www.youtube.com/watch?v=M1uyH-DzjGE&t=46s")
					elif prediction == 'Surprise':
						st.subheader("You seem to be surprised :surprise: today ! ")
						st.text("Here is your Recommended video to watch:")
						st.video("https://www.youtube.com/watch?v=M1uyH-DzjGE&t=46s")
				
			# elif feature_choice == 'Cartonize':
			# 	result_img = cartonize_image(our_image)
			# 	st.image(result_img)

			# elif feature_choice == 'Cannize':
			# 	result_canny = cannize_image(our_image)
			# 	st.image(result_canny)
	

	elif choice == 'About':
		st.subheader("About Face Expression Detection App")
		
main()	
