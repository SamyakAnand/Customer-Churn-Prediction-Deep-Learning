
# 🌀 Customer Churn Prediction Dashboard

This project is an interactive **Streamlit app** to predict customer churn in a banking dataset.
It combines a trained deep learning model, preprocessing pipeline, and user-friendly UI to estimate churn probability and help understand customer risk.

---
## ✅ Demo

*(https://github.com/SamyakAnand/Customer-Churn-Prediction-Deep-Learning/blob/main/images/Screenshot%202025-07-15%20192007.png)*

---
## 🚀 Features

* Predict churn for individual customers by adjusting sidebar inputs
* Live probability visualization with progress bar and metrics
* One-hot encoding & label encoding for categorical variables
* Data scaling and preprocessing for robust predictions
* Trained deep learning model (Keras/TensorFlow)
* Web deployment-ready using Streamlit

---

## 📂 Project Structure

```
.
├── app.py                   # Main Streamlit app
├── prediction.ipynb         # Jupyter notebook for predictions & testing
├── experiments.ipynb       # Notebook for experiments & EDA
├── model.h5                # Trained Keras model (binary classifier)
├── label_encoder_gender.pkl # Saved label encoder
├── onehot_encoder_geo.pkl   # Saved one-hot encoder
├── scaler.pkl               # Saved StandardScaler
└── requirements.txt        # Python dependencies
```

---

## 🧪 How it works

1. User selects customer attributes from the sidebar.
2. Features are encoded & scaled using saved preprocessing objects.
3. Processed input is passed to the trained Keras model.
4. App displays:

   * Predicted churn probability as %
   * Visual progress bar
   * Final churn prediction status (Likely / Unlikely)

---

## 📦 Tech Stack

* **Python**
* **Streamlit** for interactive web UI
* **TensorFlow / Keras** for deep learning
* **scikit-learn** for preprocessing (encoding, scaling)
* **pandas / numpy** for data handling

---

## ▶️ Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
streamlit run app.py
```

---

## 📊 Notebooks

* `experiments.ipynb`: Data exploration & feature engineering
* `prediction.ipynb`: Testing and validating model predictions

---

## 🔗 Links


* **[LinkedIn](https://www.linkedin.com/in/samyakanand/)**

---


## 🙏 Acknowledgements

* Built for learning and demonstration.
* Inspired by end-to-end ML project best practices.

---
