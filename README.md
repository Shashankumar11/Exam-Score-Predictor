# Exam Score Predictor

A Streamlit dashboard that scores examination performance with the existing XGBoost regressor. The app loads `best_xgb_model.pkl` and the saved label encoders. It does not retrain the model.

The regressor predicts `exam_score` from these features, in this order:

1. `study_hours`
2. `class_attendance`
3. `sleep_hours`
4. `sleep_quality`
5. `study_method`
6. `facility_rating`

Categorical values are transformed with the `LabelEncoder` objects stored in the encoder pickle. `student_id` and the other unused dataset columns are not sent to the model.

## Project files

```
exam-score-predictor/
├── app.py
├── best_xgb_model.pkl
├── label_encoders.pkl
├── Exam_Score_Prediction.csv
├── requirements.txt
└── README.md
```

The encoder file in this project is `label_encoders.pkl`. The app also accepts `label_encoders(1).pkl` if that name is present instead.

## Run the app

From this folder:

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Then open the local URL Streamlit prints, usually http://localhost:8501.

## Notes

- The model pickle was created with XGBoost 3.4.1. Install that version so the booster can be read.
- The pickle was saved on Linux. On Windows, XGBoost rejects the non-portable random-number text inside the pickle. The app rewrites only that text so the same trees can be deserialized. The exam model is not refit, and the predicted score is the model's unmodified output.
- The label encoders were saved with scikit-learn 1.6.1. They load and transform correctly with scikit-learn 1.9.x.
- Dashboard statistics and charts are calculated from `Exam_Score_Prediction.csv`.
