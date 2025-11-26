# 🧠 Deepfake Image Detection (Flask Web App)

This project is a **Deepfake Image Detection Web Application** built using **Flask** and a trained **Keras/TensorFlow image classification model**.  
Users can upload an image, and the system predicts whether the image is **Real** or **Deepfake** along with the confidence score.

---

## 🚀 Features

✔ Detects **Real vs Deepfake** images  
✔ Uses a trained **deepfake_classifier.h5** model  
✔ User-friendly **web interface** using Flask + HTML  
✔ Displays prediction label and confidence percentage  
✔ Automatically saves uploaded images to the project  
✔ Lightweight and easy to deploy

---

## 📁 Project Structure

Deepfake-Detection/
│
├── templates/
│ ├── index.html # Upload page
│ └── result.html # Prediction result page
│
├── static/
│ └── uploads/
│ └── uploaded.jpg # Last uploaded image
│
├── main.py # Flask app
├── deepfake_classifier.h5 # Trained ML/DL model
├── mlmodel.ipynb # Training notebook
├── uploaded.jpg # Sample file
└── README.md
---

## 🧠 Model Used

The app loads a trained Keras model:


This model predicts:

- **Deepfake (Fake)**  
- **Real (Authentic)**  

It outputs a **probability score**, which is converted into a label and confidence.

---

## 🧪 How the App Works

The Flask app:

1. Receives an uploaded image  
2. Saves it to `static/uploads/uploaded.jpg`  
3. Preprocesses the image (resize → normalize → convert to array)  
4. Runs prediction using the loaded model  
5. Returns:
   - Label → "Deepfake" or "Real"  
   - Confidence score (%)  

---

## 🛠️ Installation & Setup

### **1️⃣ Clone the repository**


### **2️⃣ Install dependencies**


*(Note: TensorFlow installation may vary based on your system.)*

### **3️⃣ Run the Flask app**


### **4️⃣ Open in browser**


Upload an image → View the prediction.

---

## 📸 Sample Workflow

1. Open the web page  
2. Upload an image  
3. App analyzes the image  
4. Output is shown as:


---

## 🧩 Flask Code Explanation

The core logic lies in:

### 🔹 **Loading the model**

```python
model = load_model("deepfake_classifier.h5")
def prepare_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img = img.resize((224, 224))
    img = img_to_array(img) / 255.0
    return np.expand_dims(img, axis=0)
}
prediction = model.predict(img)
prob = float(prediction[0][0])

if prob > 0.5:
    label = "Deepfake"
else:
    label = "Real"
🎯 Future Improvements

Add support for video deepfake detection

Add multiple ML models for comparison


Deploy on Render / Railway / Heroku

Add custom UI with Bootstrap or TailwindCSS

👩‍💻 Author
Jiya Joshi
Deepfake Detection Project
GitHub: https://github.com/jiya346



