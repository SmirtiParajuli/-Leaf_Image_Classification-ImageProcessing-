         *************************************** Plant Species Classification GUI ********************************************

Overview:

This project is a GUI-based application designed to classify plant species using leaf images. It utilizes Python's Tkinter library for the user interface, OpenCV 
for image processing tasks, and various machine learning models (Random Forest, Logistic Regression,  Support Vector Machine (SVM) ) for classification. The application allows users to upload
a leaf image, preprocess it, select a model for classification, and view performance metrics.

Features:

 -> Image Upload: Allows users to browse and upload leaf images.
 
 -> Preprocessing: Includes enhancement, grayscale conversion, binarization, Sobel, and Canny edge detection.
 
 -> Model Selection: Provides options to choose between different machine learning models.
 
 -> Classification: Classifies the uploaded image based on the selected model.
 
 -> Metrics Display: Shows performance metrics such as accuracy, precision, recall, F1 score, specificity, and elapsed time.
 
 -> Clear All: Resets all inputs and interface settings.
 
 -> Exit: Closes the application.

Files and Directories:

 -> Main.ipynb: The main file containing the GUI code.
 
 -> model_implementation.ipynb: Functions for feature extraction, model loading, prediction, and metric calculation.
 
 -> preprocessing.ipynb: Functions for image preprocessing.
 
 -> Datapreprocessing.ipynb: Data preparation processes, including dataset handling, feature extraction, and machine learning model training and testing. Includes hyperparameter tuning and model saving (rf_model.pkl, logistic_model.pkl, svm_model.pkl).
 
 -> Train and Test Datasets Files: FeatureTrain.csv (Training dataset), FeatureTest.csv (Test dataset).

Dependencies:

 -> Python 3.x
 
 -> Tkinter
 
 -> PIL (Pillow)
 
 -> OpenCV
 
 -> NumPy
 
 -> Matplotlib
 
 -> Pandas
 
 -> Scikit-learn
 
 -> Joblib


How to Run:

Install Dependencies:

    pip install tkinter pillow opencv-python numpy matplotlib pandas scikit-learn joblib
    
Run the GUI:

Open Main.ipynb in Jupyter Notebook or Jupyter Lab and run all cells.
Alternatively, convert the notebook to a Python script and run it directly:

              jupyter nbconvert --to script Main.ipynb
              python Main.py

Using the GUI:

 -> Click "Browse Input Image" to upload a leaf image.
 
 -> Click "Pre-processing" to process the image.
 
 -> Select a machine learning model from the "Model Selection" section.
 
 -> Click "Classification" to classify the image and view results.
 
 -> Use "Clear All" to reset the interface.
 
 -> Click "Exit" to close the application.

GUI Layout:

 -> Top Frame: Contains buttons for uploading images, preprocessing, classification, clearing inputs, and exiting.
 
 -> Left Frame: Includes radio buttons for model selection.
 
 -> Image Frame: Displays input and processed images.
 
 -> Metrics Frame: Displays classification results and performance metrics.
 
 -> Image Preprocessing Steps:

Enhancement
 -> Grayscale Conversion
 
 -> Binarization
 
 -> Sobel and Canny Edge Detection
 
 -> Classification Process:

      Feature Extraction: Extract features from the segmented image.
      Model Loading: Load the selected pre-trained model.
      Prediction: Classify the species based on extracted features.
      Metrics Calculation: Calculate and display performance metrics.

Dataset Information:
Please download  the datasets using the link below 
    Source: UCI Machine Learning Repository's Leaf dataset(https://archive.ics.uci.edu/dataset/288/leaf)
  
