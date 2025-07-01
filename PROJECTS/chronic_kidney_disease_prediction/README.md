# Chronic Kidney Disease (CKD) Prediction

This project applies machine learning techniques to predict Chronic Kidney Disease (CKD) using real-world patient data. It is based on medical features such as blood pressure, serum creatinine, hemoglobin, albumin levels, and more. The goal is to assist early diagnosis using automated predictive models.

---

## 📂 Dataset

- Source: [UCI Machine Learning Repository - CKD Dataset](https://archive.ics.uci.edu/ml/datasets/chronic_kidney_disease)
- Samples: 400 patients
- Features: 24 (mixed numerical and categorical)
- Target: `classification` (ckd / notckd)

---

## 🔄 Preprocessing Steps

- Handled missing values represented by `'?'`
- Converted data types using `pandas.to_numeric` where necessary
- Imputed:
  - Numerical columns with **median**
  - Categorical columns with **mode**
- Encoded categorical labels (e.g., `ckd` = 1, `notckd` = 0)
- Feature scaling applied for model training

---

## ⚙️ Machine Learning Models Used

| Model                  | Accuracy | Comments                         |
|-----------------------|----------|----------------------------------|
| Logistic Regression    | ~96%     | Performs well on clean datasets |
| Random Forest Classifier | **98%** ✅ | Highest accuracy and robustness |
| Support Vector Machine | ~95%     | Sensitive to feature scaling    |
| Decision Tree          | ~94%     | Easy to interpret                |

✅ **Final Model Selected:** `RandomForestClassifier`

---

## 📊 Evaluation Metrics

- **Accuracy:** 98%
- **Precision, Recall, F1-Score:** All close to 1.0
- **Confusion Matrix:** Very low false negatives and false positives

---

## 📈 Visualizations

- Correlation heatmap for feature insight
- Boxplots and histograms for univariate analysis
- Pairplot and scatterplots to observe class separation
- ROC Curve for classification evaluation

---

## 🗂️ Files Included

| File/Notebook                         | Purpose                                |
|--------------------------------------|----------------------------------------|
| `chronic_kidney_disease_prediction.ipynb` | Full EDA, preprocessing, model training |
| `chronic_kidney_disease.csv`         | Cleaned dataset (from ARFF)            |
| `random_forest_classification_model.pkl` | Saved model for deployment             |
| `CKD_Model_Deployment_Deepika.ipynb` | Deployment notebook (sample prediction) |

---

##  Deployment Ready

You can load the `.sav` model in a Streamlit or Flask app for real-time CKD predictions.

```python
# Save the model as .sav
model_path = "ckd_rf_model.sav"
joblib.dump(model, model_path)

# Save scaler too for deployment use
scaler_path = "ckd_scaler.sav"
joblib.dump(scaler, scaler_path)

---

## 📝 How to Use

1. **Clone the repository**  
  ```bash
  git clone https://github.com/yourusername/chronic-kidney-disease-prediction.git
  cd chronic-kidney-disease-prediction
  ```

2. **Install dependencies**  
  ```bash
  pip install -r requirements.txt
  ```

3. **Run the notebook**  
  Open `chronic_kidney_disease_prediction.ipynb` in Jupyter Notebook or VS Code to explore EDA, preprocessing, and model training steps.

4. **Deploy the model**  
  Use the provided `.pkl` or `.sav` files (`random_forest_classification_model.pkl`, `ckd_rf_model.sav`, and `ckd_scaler.sav`) in your preferred deployment framework (e.g., Streamlit or Flask) for real-time predictions.

5. **Sample prediction**  
  See `CKD_Model_Deployment_Deepika.ipynb` for an example of loading the model and making predictions on new data.

---

##  Streamlit App

# streamlit app for Chronic Kidney Disease (CKD) prediction
# This app allows users to input clinical data and get predictions on CKD status using a pre-trained machine learning model.
 
# Building and Running Containers

Build the Docker image:
docker-compose build

Start the containers:
docker-compose up -d

(AI) deehub@deehub:~/JoinDeeHub/chronic-kidney-disease-prediction$ sudo docker-compose ps
NAME                IMAGE               COMMAND             SERVICE             CREATED             STATUS              PORTS

(AI) deehub@deehub:~/JoinDeeHub/chronic-kidney-disease-prediction$ sudo docker-compose up
[+] Running 1/0
 ⠿ Container ckd-predictor  Created                                                                                                  0.0s
Attaching to ckd-predictor
ckd-predictor  | 
ckd-predictor  | Collecting usage statistics. To deactivate, set browser.gatherUsageStats to false.
ckd-predictor  | 
ckd-predictor  | 
ckd-predictor  |   You can now view your Streamlit app in your browser.
ckd-predictor  | 
ckd-predictor  |   URL: http://0.0.0.0:8501
ckd-predictor  | 
^Z
[14]+  Stopped                 sudo docker-compose up

URL: http://0.0.0.0:8501


(AI) deehub@deehub:~/JoinDeeHub/chronic-kidney-disease-prediction$ sudo docker-compose ps
NAME                IMAGE                                       COMMAND                  SERVICE             CREATED              STATUS              PORTS
ckd-predictor       chronic-kidney-disease-prediction-ckd-app   "streamlit run src/a…"   ckd-app             About a minute ago   Up 11 seconds       0.0.0.0:8501->8501/tcp, :::8501->8501/tcp

(AI) deehub@deehub:~/JoinDeeHub/chronic-kidney-disease-prediction$ sudo docker-compose down
[+] Running 2/2
 ⠿ Container ckd-predictor                            Removed                                                                        0.5s
 ⠿ Network chronic-kidney-disease-prediction_default  Removed                                                                        0.1s

(AI) deehub@deehub:~/JoinDeeHub/chronic-kidney-disease-prediction$ sudo docker-compose ps
NAME                IMAGE               COMMAND             SERVICE             CREATED 



View application logs:
docker-compose logs -f web

Stop containers:
docker-compose down