# Breast Cancer Prediction Using Machine Learning

## Project Overview

Breast cancer is one of the leading causes of cancer-related mortality among women worldwide. While breast masses are common and often benign—especially in younger women—early and accurate diagnosis is essential for effective treatment and improved survival rates.

This project aims to enhance **breast cancer prediction** by applying a range of **Machine Learning algorithms** to diagnostic data. Both **classification** and **clustering** techniques are explored, along with **feature selection** and **hyperparameter optimization**, to identify models that deliver the most reliable predictive performance.

---

## Dataset Description

- **Number of Samples:** 569  
- **Number of Features:** 32  
- **Target Classes:**  
  - Malignant  
  - Benign  

The dataset breastcancer.csv consists of numerical features computed from digitized images of fine needle aspirates (FNA) of breast masses, commonly used for breast cancer diagnosis tasks.

---

## Methodology

### Feature Selection
To reduce dimensionality and improve learning efficiency, the following techniques are applied:
- **Principal Component Analysis (PCA)**
- **Kernel Principal Component Analysis (Kernel PCA)**

### Hyperparameter Optimization
- **Grid Search Cross-Validation** is used to identify optimal hyperparameters for each model.

### Machine Learning Models

#### Classification Algorithms
- Logistic Regression  
- Decision Tree  
- Random Forest  
- k-Nearest Neighbors (k-NN)  
- Naive Bayes  
- Gradient Boosting  
- Support Vector Machines (SVM)

#### Clustering Algorithms
- K-Means  
- Hierarchical Clustering  
- Mean Shift  

---

## Evaluation Metrics

### Classification Metrics
The performance of classification models is evaluated using:
- Training Accuracy  
- Test Accuracy  
- Recall  
- Precision  
- F1 Score  

These metrics are particularly important in medical diagnosis scenarios, where minimizing false negatives is critical.

### Clustering Metrics
The quality of clustering results is assessed using:
- Silhouette Score  
- Davies–Bouldin Index  
- Calinski–Harabasz Index  
- Adjusted Rand Index (ARI)  
- Normalized Mutual Information (NMI)

---

## Technologies Used

- **Programming Language:** Python  
- **Libraries & Frameworks:**  
  - NumPy  
  - Pandas  
  - Scikit-learn  
  - Matplotlib  
  - Seaborn  

---

## Conclusion

This project demonstrates the effectiveness of Machine Learning techniques in supporting early breast cancer diagnosis. By combining dimensionality reduction, hyperparameter tuning, and a diverse set of learning algorithms, the study identifies models capable of delivering high predictive accuracy. The results highlight the potential of data-driven approaches to assist medical professionals in making informed diagnostic decisions.
