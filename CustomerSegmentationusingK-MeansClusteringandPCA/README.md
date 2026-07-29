# Assignment 7 — Customer Segmentation using K-Means Clustering and PCA

## 📌 Objective
Segment mall customers into distinct groups based on their age, annual income, and spending
behavior using **K-Means Clustering**, and use **Principal Component Analysis (PCA)** to
visualize these clusters in two dimensions.

## 📊 Dataset
**Mall Customer Segmentation Dataset** (200 rows, 5 columns)
Source: [Kaggle — vjchoudhary7/customer-segmentation-tutorial-in-python](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python)

*(The dataset is not uploaded to this repo per assignment instructions — use the Kaggle link above, or the notebook loads it automatically from a public mirror.)*

## 🛠️ Libraries Used
- `pandas` — data loading and manipulation
- `numpy` — numerical operations
- `scikit-learn` — feature scaling, `KMeans`, `PCA`
- `matplotlib` — visualization (elbow curve, cluster scatter plots, PCA visualization)

## 🔍 Methodology
1. **Data Understanding** — loaded the dataset, identified numerical features (`Age`,
   `Annual Income (k$)`, `Spending Score (1-100)`) and the categorical feature (`Genre`).
2. **Data Preprocessing** — checked for missing values (none found), dropped the
   non-predictive `CustomerID` column, encoded `Genre` (Male→1, Female→0), and standardized
   all features using `StandardScaler`.
3. **Model Development** — used the **Elbow Method** across K=1 to K=10 to determine the
   optimal number of clusters (K=5), trained the final `KMeans` model, assigned cluster
   labels to each customer, and applied **PCA** to reduce the 4 standardized features to 2
   principal components.
4. **Visualization & Evaluation** — plotted the elbow curve, a scatter plot of clusters using
   Income vs Spending Score, and a PCA-based 2D visualization of all clusters, then profiled
   each cluster's average characteristics.

## 📈 Results

| Cluster | Avg Age | Avg Income (k$) | Avg Spending Score | Size | Profile |
|---------|---------|-------------------|----------------------|------|---------|
| 0 | 46.3 | 26.8 | 18.4 | 20 | Cautious / low-value |
| 1 | 25.2 | 41.1 | 62.2 | 54 | Young big spenders |
| 2 | 32.9 | 86.1 | 81.5 | 40 | Premium target customers |
| 3 | 39.9 | 86.1 | 19.4 | 39 | High-income savers |
| 4 | 55.6 | 54.4 | 48.9 | 47 | Average / general segment |

**PCA:** 2 principal components retained ≈ 77.6% of total variance across the 4 standardized
features, enough to cleanly visualize the 5-cluster structure in 2D.

## ✅ Conclusion
This project applied K-Means Clustering to segment mall customers into five distinct groups
based on their age, income, spending behavior, and gender, using the Elbow Method to
determine the optimal number of clusters and PCA to visualize the results in two dimensions.
The five segments identified — ranging from young high-spenders and premium high-income
customers to cautious low-income shoppers and high-income savers — offer clear, actionable
groups for targeted marketing campaigns, such as premium offers for high-income high-spenders
or engagement incentives for high-income customers who currently spend little. This kind of
segmentation has direct business applications in personalized promotions, loyalty program
design, and inventory/merchandising decisions tailored to each segment's preferences. One key
limitation of K-Means is that it requires the number of clusters (K) to be specified in
advance, and it assumes clusters are roughly spherical and similarly sized, which may not
reflect the true, more irregular shape of real customer segments. One key advantage of PCA is
its ability to compress multiple correlated features into a smaller number of components while
retaining most of the original variance, making it much easier to visualize and interpret
patterns in data that would otherwise have too many dimensions to plot directly.

## 📂 Files
- `Assignment-7.ipynb` — full notebook with code, outputs, and visualizations
