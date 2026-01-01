# Multilayer Perceptron (MLP) for Iris Classification

![Python](https://img.shields.io/badge/Python-3.x-blue?style=flat&logo=python)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-orange?style=flat&logo=scikit-learn)
![Dataset](https://img.shields.io/badge/Dataset-Iris-green)

##  Project Overview
This project demonstrates the implementation of a **Multilayer Perceptron (MLP)**, a class of feedforward artificial neural networks, using the **Scikit-Learn** library. The goal is to classify Iris flowers into three species (*Setosa, Versicolor, and Virginica*) based on their morphological features.

Unlike Deep Learning frameworks (like TensorFlow), using Scikit-Learn for neural networks is excellent for tabular data and demonstrates a strong grasp of fundamental machine learning concepts.

## Model Pipeline
The project follows a standard robust machine learning pipeline:

1.  **Data Loading:** Utilizing the classic Iris dataset (150 samples, 4 features).
2.  **Preprocessing:**
    * **Train-Test Split:** Separating data to ensure unbiased evaluation.
    * **Feature Scaling:** Using `StandardScaler` to normalize features (Mean=0, Variance=1), which is critical for Neural Network convergence.
3.  **Model Architecture:**
    * **Classifier:** `MLPClassifier` from `sklearn.neural_network`.
    * **Optimization:** Uses the Adam solver for weight optimization.
    * **Activation:** Uses ReLU (Rectified Linear Unit) for non-linearity.
4.  **Evaluation:** Performance measured using Accuracy Score and Classification Report (Precision, Recall, F1-Score).

##  Dataset
* **Source:** Scikit-Learn built-in datasets.
* **Features:** Sepal Length, Sepal Width, Petal Length, Petal Width.
* **Classes:** 3 (Iris-Setosa, Iris-Versicolor, Iris-Virginica).

##  Tech Stack
* **Language:** Python
* **Library:** Scikit-Learn (sklearn)
* **Visualization:** Matplotlib
* **Utilities:** NumPy, Pandas

##  Key Results
The model achieves high accuracy on the test set, demonstrating the effectiveness of even simple neural architectures on structured tabular data.

*(Note: Detailed classification metrics are available in the notebook execution outputs.)*

## How to Run
1.  **Clone the repository:**
    ```bash
    git clone [https://PyPro2024/MLP-Classifier-Iris-Sklearn.git]
    ```
2.  **Install dependencies:**
    ```bash
    pip install scikit-learn matplotlib numpy
    ```
3.  **Run the Notebook:**
    Open `MLP.ipynb` in Jupyter Notebook or Google Colab to execute the pipeline.

---
*If you find this project helpful, feel free to ⭐ the repo!*
