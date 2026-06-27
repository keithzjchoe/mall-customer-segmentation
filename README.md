# Mall Customer Segmentation using K-Means Clustering

An individual data science project applying unsupervised machine learning to segment mall customers into distinct groups based on age, annual income, and spending behaviour.

---

## Project Overview

Understanding customer behaviour is critical for targeted marketing. This project uses **K-Means clustering** to identify natural groupings within a mall's customer base, enabling data-driven insights into spending patterns across different demographic segments.

The analysis covers the full data science pipeline from data cleaning and preprocessing through to cluster evaluation, visualisation, and customer profiling.

---

## Dataset

**Source:** [Mall Customer Segmentation Data](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python) - Kaggle

| Feature | Description |
|---|---|
| `CustomerID` | Unique customer identifier |
| `Gender` | Customer gender (Male/Female) |
| `Age` | Customer age |
| `Annual Income (k$)` | Annual income in thousands of dollars |
| `Spending Score (1-100)` | Mall-assigned score based on spending behaviour |

- **Total records:** 200 customers

---

## Methodology

### 1. Data Preprocessing
- Checked and handled missing values (numeric → median, categorical → mode)
- Encoded categorical variables using Label Encoding
- Applied **manual Min-Max normalisation** on selected features to scale data between 0 and 1

### 2. Feature Selection
Clustered on three key features:
- Age
- Annual Income (k$)
- Spending Score (1-100)

### 3. Optimal K Selection
Two methods were used to determine the best number of clusters:
- **Elbow Method** : plotted inertia across k = 2 to 10
- **Silhouette Score** : selected k with the highest score across the same range

Both methods converged on **k = 5** as the optimal number of clusters.

### 4. Clustering & Evaluation
- Fitted K-Means (k=5) using `k-means++` initialisation with 10 random starts
- Evaluated using two metrics:

| Metric | Score | Interpretation |
|---|---|---|
| Silhouette Score | 0.406 | Moderate — expected for overlapping real-world behavioural data |
| Davies-Bouldin Index | 0.880 | Good — under 1.0 indicates well-separated clusters |

---

## Results

5 distinct customer segments were identified:

| Cluster | Profile |
|---|---|
| 0 | Middle-aged, moderate income, moderate spenders |
| 1 | Young, low income, low spenders |
| 2 | Young, high income, high spenders |
| 3 | Older, high income, low spenders |
| 4 | Middle-aged, low income, high spenders |

Visualisations include:
- Income vs Spending Score scatter plot (coloured by cluster)
- Age vs Spending Score scatter plot (coloured by cluster)
- Cluster profile table (mean age, income, spending per cluster)
- Gender distribution per cluster

---

## Tech Stack

- Python 3
- pandas
- NumPy
- scikit-learn (KMeans, PCA, StandardScaler, silhouette\_score, davies\_bouldin\_score)
- Matplotlib
- Google Colab

---

## How to Run

1. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python) and save it as `Mall_Customers.csv` in the same directory as the notebook
2. Open the notebook in [Google Colab](https://colab.research.google.com/) or Jupyter Notebook
3. Run all cells in order

---

## Acknowledgements

Individual assignment : Taylor's University, Bachelor of Computer Science (Data Science minor).  
Dataset sourced from Kaggle ([vjchoudhary7](https://www.kaggle.com/vjchoudhary7)).
