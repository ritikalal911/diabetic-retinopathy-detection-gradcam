# Diabetic Retinopathy Severity Classification with EfficientNet-B3 and Grad-CAM

## Overview

This project develops an end-to-end deep learning pipeline for **Diabetic Retinopathy (DR) severity classification** using retinal fundus images from the **APTOS 2019 Blindness Detection dataset**.

The system classifies retinal images into five diabetic retinopathy severity levels:

| Class | Severity         |
| ----- | ---------------- |
| 0     | No DR            |
| 1     | Mild             |
| 2     | Moderate         |
| 3     | Severe           |
| 4     | Proliferative DR |

The final pipeline covers exploratory data analysis, image preprocessing, class imbalance handling, EfficientNet-B3 model training, model evaluation, and Grad-CAM explainability.

---

## Project Objectives

The main objectives of this project are to:

* Explore the characteristics and class distribution of the APTOS 2019 dataset.
* Develop a consistent preprocessing pipeline for retinal fundus images.
* Address the significant class imbalance present in the dataset.
* Train an EfficientNet-B3 deep learning model for five-class DR severity classification.
* Evaluate model performance using classification and ordinal classification metrics.
* Apply Grad-CAM to investigate which retinal regions influence model predictions.

---

## Dataset

The project uses the **APTOS 2019 Blindness Detection** dataset, which contains retinal fundus photographs labelled according to diabetic retinopathy severity.

Due to the size of the image dataset, the raw and generated image files are not stored directly in this repository.

The dataset can be obtained from the APTOS 2019 Blindness Detection competition on Kaggle.

---

## Project Pipeline

The final implementation is organized into five notebooks.

### 1. Exploratory Data Analysis

`01_eda.ipynb`

Explores the dataset before model development, including:

* Class distribution
* Class imbalance
* Image dimensions and resolution
* Image brightness and contrast
* Sample retinal images from each severity class
* Image quality characteristics

The results of the EDA were used to guide preprocessing and model-development decisions.

### 2. Image Preprocessing

`02_preprocessing.ipynb`

Implements the preprocessing pipeline used to prepare retinal images for model training.

The preprocessing stage includes image preparation, resizing and transformation of the retinal images into a consistent format suitable for EfficientNet-B3.

Data augmentation is also incorporated into the training pipeline to improve model generalization.

### 3. Model Training

`03_model_training.ipynb`

Implements the deep learning training pipeline using **EfficientNet-B3**.

The model uses transfer learning to leverage representations learned from ImageNet while adapting the network to five-class diabetic retinopathy severity classification.

The training pipeline also incorporates techniques for handling the imbalanced class distribution.

### 4. Model Evaluation

`04_evaluation.ipynb`

Evaluates the trained model on unseen test data.

Evaluation outputs include:

* Classification report
* Confusion matrix
* Normalized confusion matrix
* Test predictions
* Class-wise performance
* Error-distance analysis
* Evaluation summary and metrics

Because DR severity is ordinal, evaluation is not limited to simple classification accuracy. The analysis also considers how far incorrect predictions are from the true severity grade.

### 5. Grad-CAM Explainability

`05_gradcam_explainability.ipynb`

Uses **Gradient-weighted Class Activation Mapping (Grad-CAM)** to examine the regions of retinal images that influence model predictions.

Grad-CAM visualizations are generated for:

* Correct predictions across DR severity grades
* Incorrect predictions
* Selected difficult cases

This provides an additional layer of interpretability beyond numerical performance metrics and helps investigate whether the model is focusing on meaningful retinal regions when making predictions.

---

## Repository Structure

```text
diabetic-retinopathy-detection-gradcam/
│
├── 01_eda.ipynb
├── 02_preprocessing.ipynb
├── 03_model_training.ipynb
├── 04_evaluation.ipynb
├── 05_gradcam_explainability.ipynb
│
├── artifacts/
│   ├── preprocessing/
│   └── results/
│
├── data/
│   ├── eda/
│   └── splits/
│
├── studies/
│   ├── 01_eda/
│   │   └── figures/
│   ├── 02_preprocessing/
│   │   └── figures/
│   ├── 04_evaluation/
│   │   └── figures/
│   └── 05_gradcam/
│       └── figures/
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Individual Research and Contributions

Before integrating the final pipeline, team members independently investigated different components of the project. These experiments helped inform decisions used in the final implementation.

### Ritika Lal — EDA and Image Preprocessing

**Research areas:**

* Exploratory Data Analysis
* Retinal image characteristics
* Image preprocessing
* Preprocessing experiments
* Data preparation

Individual research repository:

**[Explainable DR Severity Grading – Ritika Lal](https://github.com/ritikalal911/explainable-dr-severity-grading)**

### Peter Bello — Model Training

**Research areas:**

* Model architecture investigation
* Transfer learning
* Model training strategies
* Training configuration and experimentation

**Individual Research Repository:**
*Peter's repository*

### Alain Dika — Class Imbalance and Explainability

**Research areas:**

* Analysis of class imbalance
* Strategies for handling underrepresented DR severity classes
* Explainability techniques

**Individual Research Repository:**
*Alain's repository*

---

## Final Group Integration

The individual research conducted by each team member was used to inform the final group implementation.

The five notebooks located in the root of this repository represent the **final integrated pipeline** rather than separate individual experiments.

```text
EDA
 ↓
Preprocessing
 ↓
Class Imbalance Handling
 ↓
EfficientNet-B3 Training
 ↓
Evaluation
 ↓
Grad-CAM Explainability
```

This separation allows the repository to preserve a clean final implementation while individual repositories document the experimentation and research that contributed to the final design.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/ritikalal911/diabetic-retinopathy-detection-gradcam.git
cd diabetic-retinopathy-detection-gradcam
```

Install the required Python packages:

```bash
pip install -r requirements.txt
```

### Main Dependencies

* PyTorch
* Torchvision
* NumPy
* Pandas
* Scikit-learn
* Matplotlib
* Seaborn
* OpenCV
* Pillow
* tqdm
* Jupyter

---

## Running the Project

The notebooks should be executed in the following order:

```text
01_eda.ipynb
      ↓
02_preprocessing.ipynb
      ↓
03_model_training.ipynb
      ↓
04_evaluation.ipynb
      ↓
05_gradcam_explainability.ipynb
```

The raw APTOS dataset must be downloaded separately before running the complete pipeline.

---

## Generated Outputs

The project produces several artifacts during preprocessing, training and evaluation.

Examples include:

```text
artifacts/
├── preprocessing/
│   ├── aptos_preprocessed_384_manifest.csv
│   └── class_weights_effective_number.csv
│
└── results/
    ├── classification_report.csv
    ├── efficientnet_b3_training_history.csv
    ├── evaluation_summary.json
    ├── test_metrics.csv
    ├── test_metrics.json
    └── test_predictions.csv
```

Visual outputs include:

* Class distribution plots
* Image characteristic analysis
* Raw vs. preprocessed image comparisons
* Augmentation examples
* Confusion matrices
* Prediction error analysis
* Grad-CAM heatmaps
* Grad-CAM analysis of misclassified images

---

## Technologies Used

**Programming Language**

* Python

**Deep Learning**

* PyTorch
* Torchvision
* EfficientNet-B3
* Transfer Learning

**Computer Vision**

* OpenCV
* Pillow
* Grad-CAM

**Data Analysis**

* NumPy
* Pandas

**Machine Learning & Evaluation**

* Scikit-learn

**Visualization**

* Matplotlib
* Seaborn

---

## Team Members

| Team Member     | Primary Research Area            |
| --------------- | -------------------------------- |
| **Ritika Lal**  | EDA & Image Preprocessing        |
| **Peter Bello** | Model Training                   |
| **Alain Dika**  | Class Imbalance & Explainability |

All team members contributed to the integration and development of the final project pipeline.

---

## Disclaimer

This project was developed for **academic and research purposes**.

The model is not a medical diagnostic system and should not be used to make clinical decisions. Model predictions and Grad-CAM visualizations are intended only for experimentation and analysis of deep learning methods for diabetic retinopathy severity classification.
