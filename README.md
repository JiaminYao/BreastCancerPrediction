# 🩺 Breast Cancer Prediction Using Machine Learning

## 📌 Project Overview

Breast cancer is one of the leading causes of cancer-related deaths among women globally. Although breast masses are common and often benign, especially in younger women, **early and accurate diagnosis** is critical for effective treatment and improved survival rates.

This project applies **Machine Learning techniques** to enhance the prediction of breast cancer using diagnostic data. Both **supervised classification** and **unsupervised clustering** approaches are explored, combined with **feature reduction** and **hyperparameter tuning**, to achieve optimal predictive performance.

## 📊 Dataset Description

- **Total Samples:** 569  
- **Total Features:** 32  
- **Target Classes:**  
  - Malignant  
  - Benign  

The dataset contains numerical features extracted from digitized images of **fine needle aspirates (FNA)** of breast tissue, widely used in breast cancer diagnostic research.

## 🧠 Methodology

### 🔍 Feature Selection
To reduce dimensionality and improve model generalization:
- **Principal Component Analysis (PCA)**
- **Kernel Principal Component Analysis (Kernel PCA)**

### ⚙️ Hyperparameter Optimization
- **Grid Search Cross-Validation** is employed to determine optimal model parameters.

### 🤖 Machine Learning Models

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

## 📈 Evaluation Metrics

### 📋 Classification Metrics
Used to assess supervised learning performance:
- Training Accuracy  
- Test Accuracy  
- Recall  
- Precision  
- F1 Score  

These metrics are essential in medical diagnosis, where reducing **false negatives** is especially important.

### 🧮 Clustering Metrics
Used to evaluate unsupervised learning quality:
- Silhouette Score  
- Davies–Bouldin Index  
- Calinski–Harabasz Index  
- Adjusted Rand Index (ARI)  
- Normalized Mutual Information (NMI)


## 🛠️ Technologies Used

- **Python**
- **Libraries**
  - NumPy  
  - Pandas  
  - Scikit-learn  
  - Matplotlib  
  - Seaborn  


## 🎯 Conclusion

This project highlights the potential of **Machine Learning** in assisting early breast cancer diagnosis. By integrating dimensionality reduction, hyperparameter optimization, and multiple learning algorithms, the study identifies models capable of delivering high predictive accuracy. These results demonstrate how data-driven solutions can support medical professionals in making informed diagnostic decisions.

