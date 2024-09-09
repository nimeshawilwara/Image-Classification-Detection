Image Classification App

Welcome to the Image Classification Application. This is a web application in which any user can classify an image by making use of the pre-trained InceptionV3 model. The application is created using Streamlit and TensorFlow for effective and interactive deep learning-based image classification.

Features
• supported format for uploading images: JPG, JPEG, PNG
• Image classification based on the pre-trained InceptionV3 model on the ImageNet dataset.
- Shows the top predictions of labels along with confidence scores.

  Usage
1. In order to launch the app, follow the installation steps described above.
2. Click the sidebar and read a short intro description about the app.
3. Upload an image using the "Select an Your Image." uploader.
4. The app will depict to you the uploaded image and classify it by using the InceptionV3 model.
5. Observe the highest-scoring classification results, along with the confidence scores of the predictions.

Installation

Follow these steps to run the application locally:

git clone https://github.com/your-username/image-classification-app.git
cd image-classification-app

Create a virtual environment and activate it:

python -m venv venv
source venv/bin/activate   # On Windows use `venv\Scripts\activate`

Install dependencies:

pip install -r requirements.txt

Run the app:

streamlit run app.py
