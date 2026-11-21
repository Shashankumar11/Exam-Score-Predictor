# 🎓 Exam Score Predictor

<p align="center">

  **A Machine Learning-Based Exam Score Prediction Application**

  Predict student examination performance using study habits and academic factors with an XGBoost Regression model.

  [🚀 Live Demo](https://exam-score-predictor-ko6zjguhpt6g2q3vwe2dka.streamlit.app) · [📂 GitHub Repository](https://github.com/Shashankumar11/Exam-Score-Predictor)

</p>

---

## 📌 About The Project

Exam Score Predictor is a Machine Learning web application built using **Python, XGBoost, and Streamlit**.

The application predicts a student's expected examination score based on different academic and lifestyle-related factors.

It uses a pre-trained XGBoost Regressor model to generate predictions and provides an interactive dashboard for exploring dataset insights and predicting examination performance.

## ✨ Features

- **Score Prediction:** Predict examination scores using a trained Machine Learning model.
- **Interactive Dashboard:** Explore examination score statistics and dataset insights.
- **Dataset Visualization:** Understand score distributions and relationships between different factors.
- **Feature Analysis:** Visualize the factors influencing model predictions.
- **Categorical Encoding:** Uses saved label encoders to process categorical inputs.
- **User-Friendly Interface:** Interactive and responsive Streamlit interface.

## 📊 Model Inputs

The prediction model uses six input features:

| Feature | Description |
|---|---|
| Study Hours | Number of hours spent studying |
| Class Attendance | Student's class attendance percentage |
| Sleep Hours | Average hours of sleep |
| Sleep Quality | Self-reported sleep quality |
| Study Method | Preferred study method |
| Facility Rating | Rating of available study facilities |

## 🤖 Machine Learning Model

| Parameter | Details |
|---|---|
| Algorithm | XGBoost Regressor |
| Learning Task | Regression |
| Prediction Target | `exam_score` |
| Number of Trees | 300 |
| Maximum Depth | 3 |
| Learning Rate | 0.05 |
| Dataset Records | 20,000 |

The trained model and label encoders are saved as pickle files and loaded by the application during prediction.

## 🛠️ Tech Stack

- **Programming Language:** Python
- **Machine Learning:** XGBoost, Scikit-learn
- **Data Processing:** Pandas, NumPy
- **Data Visualization:** Plotly
- **Web Framework:** Streamlit
- **Model Storage:** Pickle
- **Deployment:** Streamlit Community Cloud
- **Version Control:** Git & GitHub

## 📁 Project Structure

```text
Exam-Score-Predictor/
│
├── .streamlit/
│   └── config.toml
│
├── app.py
├── best_xgb_model.pkl
├── label_encoders.pkl
├── Exam_Score_Prediction.csv
├── requirements.txt
├── README.md
└── .gitignore
```

## ⚙️ Installation & Setup

Follow these steps to run the project locally.

**1. Clone the repository**

```bash
git clone https://github.com/Shashankumar11/Exam-Score-Predictor.git
```

**2. Navigate to the project directory**

```bash
cd Exam-Score-Predictor
```

**3. Create a virtual environment**

```bash
python -m venv venv
```

**4. Activate the virtual environment**

Windows:

```bash
venv\Scripts\activate
```

**5. Install dependencies**

```bash
pip install -r requirements.txt
```

**6. Run the Streamlit application**

```bash
streamlit run app.py
```

The application will open in your browser at:

```text
http://localhost:8501
```

## 🚀 Live Application

Try the deployed application here:

### [Open Exam Score Predictor →](https://exam-score-predictor-ko6zjguhpt6g2q3vwe2dka.streamlit.app)

## 🔮 Future Improvements

- Improve prediction performance through further model experimentation.
- Add additional data visualizations and analytical insights.
- Introduce model evaluation metrics and performance comparisons.
- Enhance the user experience with additional interactive features.

## 👨‍💻 Author

**Shashank Kumar**

- GitHub: [@Shashankumar11](https://github.com/Shashankumar11)

---

<p align="center">
  ⭐ If you find this project interesting, consider giving it a star!
</p>

<p align="center">
  **Built with Python, Machine Learning & Streamlit**
</p>