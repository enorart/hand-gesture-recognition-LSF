# hand-gesture-recognition-LSF
Using OpenCV, mediapipe and tensorflow, I created with Python a sample program to recognize some sign in the French Sign Language (LSF)
Estimate hand pose using MediaPipe (Python version)
Capture in real time with the webcam using OpenCV
Made a neural network with tensorflow, and using tensorflow lite.

<br> ❗ _️**I got inspired by this [repo](https://github.com/Kazuhito00/hand-gesture-recognition-using-mediapipe). I used the same skeleton for the tensorflow part, as I'm new to Machine learning**_ ❗ <br>

<br> 🚧 _️**WORK IN PROGRESS. The current version can recognize only some sign (almost all the letters of the LSF alphabet). I might make updates by adding other signs (numbers, usuals signs) if I have the time and the ability to do so. For the moment the model only classifies fixed signs, without particular movements, but I would like to switch to a model that recognizes certain dynamic signs. Feel free to review ! Do not hesitate to participate in the project if you wish (improvement of the model, data collection) ! Feel free to use my project as a basis for yours !**_ 🚧 <br>

<br> 😓 _️**Sorry for my bad English, it's not my native language. I might do an English translation if many of you ask for it. The actual version is in French, but it's not very important to understand the code**_ 😓 <br>

https://github.com/enorart/hand-gesture-recognition-LSF/assets/135878234/ee9ab4e6-2a5e-443c-8be2-ed58c062d536

This repository contains the following contents.

* The main program (main.py)
* Hand sign recognition model (TFLite)
* Learning data (CSV) for hand sign recognition and notebook for learning

# Requirements
* mediapipe 0.8.1 or Later
* OpenCV 3.4.2 or Later
* Tensorflow 2.3.0 or Later
* scikit-learn 0.23.2 or Later
* Numpy 1.24.2 or Later

# Demo 
You need to run the _**main.py**_ file in your Python environment (for example VScode, Pycharm, Spyder...)

# Directory
<pre>
│  main.py
│  
├─model
│  │  dataset.csv
│  │  classification.ipynb
│  │  classification.hdf5
│  │  classification.py
│  │  classification.tflite
│  └─ label.csv
</pre> 

### main.py
This is the main program for inference.
Real time classification of sign via your webcam
In addition, learning data (key points) for hand sign recognition via the training mode (more info below)

### classification.ipynb
This is a model training script for hand sign recognition. Current model work for static sign.

### dataset.csv
Training data : normalized landmarks of the hand, for 21 sign (almost the entire alphabet, the missing letters are : J P X Y Z, because these are signs with an associaated movement)

### label.csv 
To link the identifier of the sign to its name. For the training, when the program ask you in the python console to type the id of the sign, you need to type : _**number of the row of the sign - 1**_

### classification.hdf5
Trained model for the data in the _**dataset.csv**_ file

### classification.tflite
Trained model converted to a tensorflowlite model

# Training 
### 1.Data collection
The model improves all the more as the volume of data is large and varied. <br>
Press "k" to enter in the _**Training Mode**_ (displayed as 「Mode Apprentissage」) <br>
A message in the python console appears asking you which sign do you want to register : you need to type the id of the sign <br>
After, you need to press "p" to save the normalized landmarks of your position <br> 
You can press "l" to return in the _**Normal Mode**_ <br> <br>
Note : you can add more signs or new signs by updating the _**label.csv**_ file and delete the existing data of the _**dataset.csv**_ file <br>

### 2.Model training
You need to retrain the model to use the new data from the dataset. <br>
Open "[model/classification.ipynb](model/classification.ipynb)" in Jupyter Notebook and execute from top to bottom.<br>
To change the number of training data classes, change the value of "NUM_CLASSES = 21" <br>and modify the label of "model/label.csv" as appropriate.<br><br>

# Reference
* [MediaPipe](https://mediapipe.dev/)
* [TensorFlow](https://www.tensorflow.org/)

# Author
Eno [https://github.com/enorart](https://github.com/enorart)

# License
hand-gesture-recognition-LSF is under [Apache v2 license](LICENSE)




