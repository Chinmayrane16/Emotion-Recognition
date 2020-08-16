# Emotion-Recognition Web Application With Streamlit 
A CNN based Tensorflow implementation on facial expression recognition (FER2013 dataset), achieving 66,72% accuracy 
![](images/model.png)

### Dependencies:
- python 3.7<br/>
- Keras with TensorFlow as backend<br/>
- Streamlit framework for web implementation

### FER2013 Dataset:
- Dataset from https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data<br/>
- Image Properties: 48 x 48 pixels (2304 bytes)<br/>
- labels: 0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral<br/>
- The training set consists of 28,708 examples.<br/>
- The model is represented as a json file :model.json

The separated dataset is already available to download in the two folders train and test.
### Run the project on your local:

- Open Anaconda's command prompt on the project's directory.<br/>
- Install virtual environment : `pip install virtualenv` </br>
- Create virtual environment : `virutalenv venv` </br>
- Activate the virtual environment : `source venv/bin/activate` </br>
- Install dependencies requirements.txt : `pip install -r requirements.txt`</br>
- Install Streamlit : `pip install streamlit` <br/>
- Run the app.py file : `streamlit run app.py"`

### Demos:
![](images/demo.PNG)

