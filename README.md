# Student Performance Predictor

An end-to-end machine learning project to predict student performance based on demographic and academic features. This project demonstrates the full ML workflow: data ingestion, preprocessing, model training, evaluation, and deployment as a web application using Flask.

## Table of Contents

- [Project Overview](#project-overview)
- [Project Structure](#project-structure)
- [Setup Instructions](#setup-instructions)
- [Usage](#usage)
- [Model Details](#model-details)
- [Docker Usage](#docker-usage)
- [Notebooks](#notebooks)
- [License](#license)

---

## Project Overview

This project predicts student scores using various features such as gender, ethnicity, parental education, lunch type, and test preparation course. It includes:

- Data exploration and visualization
- Data preprocessing and transformation
- Model training (CatBoost, XGBoost, etc.)
- Model evaluation
- Web app for predictions

---

## Project Structure

```
.
├── app.py                  # Flask web application
├── Dockerfile              # Docker configuration
├── requirements.txt        # Python dependencies
├── setup.py                # Package setup
├── README.md               # Project documentation
├── artifacts/              # Saved models and data
│   ├── model.pkl
│   ├── preprocessor.pkl
│   ├── raw.csv
│   ├── test.csv
│   └── train.csv
├── catboost_info/          # CatBoost training logs
├── notebook/               # Jupyter notebooks for EDA and training
│   ├── 1 . EDA STUDENT PERFORMANCE .ipynb
│   ├── 2. MODEL TRAINING.ipynb
│   └── data/
│       └── stud.csv
├── source/                 # Source code
│   ├── exception.py
│   ├── logger.py
│   ├── utils.py
│   ├── components/
│   │   ├── data_ingestion.py
│   │   ├── data_transformation.py
│   │   └── model_trainer.py
│   └── pipeline/
│       ├── predict_pipeline.py
│       └── train_pipeline.py
├── templates/              # HTML templates for Flask
│   ├── index.html
│   └── pred.html
```

---

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/sujalkyal/Student-performance-predictor.git
cd Student-performance-predictor
```

### 2. Create and Activate a Python Environment

```bash
python -m venv venv
venv\Scripts\activate   # On Windows
# Or
source venv/bin/activate   # On Linux/Mac
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Train the Model (Optional)

If you want to retrain the model, use the scripts in `source/pipeline/train_pipeline.py` or the Jupyter notebooks in `notebook/`.

---

## Usage

### 1. Run the Flask App

```bash
python app.py
```

The app will be available at `http://0.0.0.0:5000/` (or `http://localhost:5000/`).

### 2. Web Interface

- **Home Page:** `/` — Introduction and navigation.
- **Prediction Page:** `/predictdata` — Enter student details to get predicted scores.

---

## Model Details

- **Features Used:** Gender, Ethnicity, Parental Level of Education, Lunch, Test Preparation Course, Reading Score, Writing Score.
- **Algorithms:** CatBoost, XGBoost, and others (see notebooks and `model_trainer.py`).
- **Artifacts:** Trained model and preprocessor are saved in `artifacts/`.

---

## Docker Usage

To run the project in a Docker container:

### 1. Build the Docker Image

```bash
docker build -t student-performance-predictor .
```

### 2. Run the Docker Container

```bash
docker run -p 5000:5000 student-performance-predictor
```

---

## Notebooks

- **EDA:** `notebook/1 . EDA STUDENT PERFORMANCE .ipynb`
- **Model Training:** `notebook/2. MODEL TRAINING.ipynb`

These notebooks provide step-by-step data analysis and model development.

---

## License

This project is for educational purposes.

---

**Author:** Sujal Kyal  
**Contact:** sujalkyal2704@gmail.com
**Website:** [sujalkyal.dev.in](https://sujaldev-ten.vercel.app/)

---

If you have any questions or issues, please open an issue on the repository.