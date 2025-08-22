🚀 CIFAR-10 Image Classification with CNN

This project implements a Convolutional Neural Network (CNN) trained on the CIFAR-10 dataset to classify images into 10 categories:
airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
It also includes an interactive Streamlit app to test the model on random test images or your own uploaded images.


📂 Project Structure

.
├── app.py                       # Streamlit app
├── cifar-10-cnn-project.ipynb   # Project code
├── model.h5                     # Trained CNN model (saved after training)
├── requirements.txt             # Dependencies
└── README.md                    # Project documentation
 

⚙️ Installation & Setup
Clone the repo:
git clone https://github.com/your-username/cifar10-cnn.git
cd cifar10-cnn

Install dependencies:
pip install -r requirements.txt

(Optional) Train the model:
# inside your training script / notebook
model.save("cnn_cifar10.h5")

Run the Streamlit app:
streamlit run app.py

🖼️ Usage
Random Test Image → Select a random CIFAR-10 test image and predict its class.
Upload Your Own Image → Upload an image (resized to 32x32) and get prediction probabilities across all classes.


📊 Model Performance
Dataset: CIFAR-10 (60,000 32x32 color images, 10 classes)
Model: CNN with Conv2D, MaxPooling, Dropout, and Dense layers
Accuracy: ~88% (as per your results ✅)


🔧 Requirements
See requirements.txt
:
streamlit
tensorflow
numpy
matplotlib
Pillow


✨ Future Improvements
Add training history visualization (loss/accuracy curves)
Improve accuracy with data augmentation or deeper CNNs
Deploy on HuggingFace Spaces / Streamlit Cloud


📌 Author
👤 Shaikh Abdul Wahid
