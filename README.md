# 📈 M-A Predictor (Machine Learning Predictor Web App)

This is a lightweight, Streamlit-based web application that uses machine learning to perform real-time predictions on your data. It is containerized and/or virtualized using Python 3.11 to ensure compatibility with TensorFlow and other key dependencies.

---

## 🚀 Features

- ✅ Streamlit interface for interactive user input
- ✅ Real-time predictions using a trained ML model
- ✅ Compatible with Python 3.11
- ✅ Virtual environment / Docker ready
- ✅ Easy setup and isolated from system-wide Python

---

## 🧱 Tech Stack

| Component       | Version         |
|----------------|-----------------|
| Python         | 3.11.x          |
| Streamlit      | >=1.33.0        |
| TensorFlow     | 2.15.0 (CPU)    |
| NumPy          | >=1.26.4        |
| Pandas         | >=2.2.2         |

---

## 📂 Folder Structure
```bash
m-apredictor/
├── app.py # Main Streamlit app
├── model/ # Saved ML model or preprocessing logic
├── requirements.txt # All Python dependencies
├── README.md # Project README (this file)
└── Dockerfile # (Optional) For containerized deployment
```


## ⚙️ Setup Instructions

### ✅ Option 1: Local (Virtualenv with Python 3.11)

> Install Python 3.11 if it's not yet on your system:
```bash
sudo apt install python3.11 python3.11-venv
```

Create a Python 3.11 virtual environment:
```bash
python3.11 -m venv streamlit-venv
source streamlit-venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Run the app:
```bash
streamlit run app.py
```

🐳 Option 2: Docker Deployment
Build and run the app using Docker:

```bash
docker build -t m-apredictor .
docker run -p 8501:8501 m-apredictor
Then visit http://localhost:8501 in your browser.
```

💡 Option 3: Using uv (Optional)
If you're using uv:

```bash
uv venv python3.11
uv pip install -r requirements.txt
uv pip streamlit run app.py
```

🔐 Python Version Compatibility
Because tensorflow-cpu==2.15.0 does not support Python 3.13, we pin the app environment to Python 3.11, isolating it from your system-wide Python version.

🧠 Model Info
Model Type: Customized MLPClassifier 

Training Data: Engineered Dataset of M&A Data

Target Variable: Acqurired

Input Format: List of input features the user must provide

📜 Example .requirements.txt
```bash
streamlit>=1.33.0
tensorflow-cpu==2.15.0
numpy>=1.26.4
pandas>=2.2.2
```

🔧 Troubleshooting
❌ tensorflow-cpu==2.15.0 not installable on Python 3.13
Solution: Use Python 3.11 via virtualenv or Docker.

❌ App crashes due to incompatible packages
Check your Python version with python --version

Check your virtualenv is activated with which python

Reinstall dependencies via pip install -r requirements.txt

📬 Contact
For questions, bugs, or improvements, feel free to open an issue or reach out to the maintainer.

📄 License
This project is licensed under the MIT License.
---
