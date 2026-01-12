# Handwritten Digit Recognition – Classical ML Pipeline

This repository contains an end‑to‑end handwritten digit recognition pipeline using classical machine learning algorithms (KNN, SVM, Decision Tree, custom KNN from scratch, Voting Ensemble, and PCA). The core implementation is in a single Jupyter notebook: `AI_assignment-Virtualyyst_ShivaSukumar.ipynb`.

---

## Dataset

- **Source**: [MNIST.csv – prasertcbs/basic-dataset](https://github.com/prasertcbs/basic-dataset/blob/master/MNIST.csv)  
- The data consists of 8×8 grayscale digit images flattened to 64 pixel values plus a target label column.  
- For this project the 8×8 images are padded to 28×28 (and flattened to 784 features) to mimic the usual MNIST format.

---

## Project Pipeline

The overall workflow follows the steps in the flowchart below, from data ingestion to final evaluation and reporting.

![Flowchart](flowchart.jpg)


### 1. Data Ingestion & EDA
- Load `MNIST.csv` into a pandas DataFrame and inspect the total number of samples.  
- Check class distribution and verify there are no missing values.  
- Visualize example digits by reshaping the 8×8 arrays and padding them to 28×28 for display.

### 2. Data Preprocessing
- **Padding**: Place each 8×8 digit image in the center of a 28×28 canvas and flatten to a 784‑dimensional vector.  
- **Normalization**: Scale pixel values from 0–255 to 0–1 by dividing by 255.0.  
- **Train–Test Split**: Use an 80% / 20% split with a fixed random state.

### 3. Model Implementation

**Scikit‑learn models:**

- K‑Nearest Neighbors (KNN) – `KNeighborsClassifier(n_neighbors=5)`  
- Support Vector Machine (SVM) – `SVC(kernel="rbf", C=1.0, gamma="scale")`  
- Decision Tree – `DecisionTreeClassifier(max_depth=10, random_state=42)`

**Manual KNN (from scratch):**

- NumPy‑based KNN that computes Euclidean distance to all training samples, selects the k nearest neighbors, and predicts the majority label using `collections.Counter`.

**Bonus implementations:**

- Hard Voting Ensemble that combines KNN, SVM, and Decision Tree.  
- PCA retaining 95% variance to reduce dimensionality from 784 to around 28 components, followed by KNN and SVM trained on the PCA features.

---

## Evaluation & Results

### Metrics

For each model the notebook computes:

- Test accuracy.  
- Confusion matrix heatmaps using `confusion_matrix` and Seaborn.  
- Visualizations of typical misclassified digits from the KNN model.

### Accuracy Summary

| Model                          | Accuracy (Original) | Accuracy (with PCA) |
|-------------------------------|---------------------|---------------------|
| K‑Nearest Neighbors (KNN)     | 98.61%              | 98.06%              |
| Support Vector Machine (SVM)  | 98.33%              | 98.61%              |
| Decision Tree                 | 86.39%              | N/A                 |
| Custom KNN (scratch)          | 98.61%              | N/A                 |
| Voting Ensemble (KNN+SVM+DT)  | 98.33%              | N/A                 |

These accuracies come from the 80/20 train–test split defined in the notebook.

### Key Observations

- The custom scratch KNN exactly matches the scikit‑learn KNN accuracy (98.61%), confirming the distance and voting implementation.  
- SVM with PCA achieves 98.61% while using far fewer features, which makes it more efficient for deployment.  
- Decision Tree performance is lower (~86%) due to its sensitivity to small spatial variations in the digit images.  
- Misclassifications mostly occur between visually similar digits like 3 vs 8 and 4 vs 9.

```text
.
├── AI_assignment-Virutalyyst_ShivaSukumar.ipynb  
├── flowchart.jpg                                 
├── requirements.txt                              
└── README.md     
