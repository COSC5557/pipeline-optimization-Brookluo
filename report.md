# Hyperparameter optimization report

## Introduction

This report is a summary of the pipeline optimization process for the models used to predict the white wine quality with `winequality-white.csv` dataset. This pipeline optimization is largely based on the previous assignment on hyperparameter optimization with additions on the preprocessing tunning. Since there are only seven different quality levels (i.e. 3-9), I treat this problem as a classification problem. The pipeline consists of three components: a scaler, a dimension reducer, a ML model. The goal is to find the best preprocessing methods and hyperparameters for the pipeline to maximize the accuracy of the predictions. Three ML models are used in this report: Random Forest, Gradient Boosting, and Support Vector Machine. The scalers are `RobustScaler`, `PowerTransformer`, `Normalizer`, `StandardScaler`, `MinMaxScaler`, and `StandardScaler`. The dimension reducers are `PCA`, `FastICA`.

## Dataset description

The white wine dataset contains 4898 observations with 12 columns in each observation. There is no missing data in the dataset. The target variable is `quality`, which is an integer between 3 and 9. The rest of the columns are features of the wine. I consider all feature columns as continuous numerical variables. The dataset is divided into 70% training and 30% testing.

## Experiment setup

Similar to the HPO assignment, I used Julia to run the experiments with the two main packages `AutoMLPipeline` and `Hyperopt` to perform the hyperparameter optimization. A random seed is used to ensure experiment reproducibility. Given the type of the problem, I use accuracy as the metric to evaluate the model performance.

Nested resampling is used in the pipeline HPO process. A outer sampling is performed to split the dataset into 70% training and 30% testing. The inner sampling is performed with a 5-fold cross-validation to evaluate the model performance. The outer sampling is used to evaluate the model performance with the best hyperparameters found in the inner sampling.

I will discuss the experiment setup in two sections: preprocessing and hyperparameter optimization.

### Preprocessing

The preprocessing consists of two steps: scaling and dimension reduction. The scaling is performed before the dimension reduction. The scaling is performed with one of the six scalers aforementioned. The dimension reduction is performed with one of the two dimension reducers with number of components as a hyperparameter to be optimized. The number of components for PCA/FastICA is in the range [2, 20). The pipeline optimization process for preprocessing is to find the best combination of scaler, dimension reducer, and number of components to maximize the accuracy of the model.


### Hyperparameter optimization

The hyperparameter optimization is performed using the `Hyperopt` package using Hyperband with random sampler (RSHB). The maximum number of iterations is set to 50 for random sampler, and the resource for HB is R=50 with n=3. The best hyperparameters are selected based on the best accuracy.

## Results

### Random Forest

The hyperparameters to optimize are number of estimators with range [10, 300), max tree depth [1, 30) and max number of features [1, 30). The best hyperparameters are shown in the table below.

5-fold test CV mean accuracy | n_estimators | max_depth | max_features | dim reducer | n_component | scaler| 
|---------|-------------|------------------|-------------|-------------|-------------|----------|
|  0.5997 | 139 | 23 | 16 | PCA | 6 | PowerTransformer |

### Gradient Boosting

The hyperparameters to optimize are number of estimators with range [10, 500), learning rate [0.01, 0.5) and max depth [1, 30).

The best hyperparameters are shown in the table below.

| 5-fold test CV mean accuracy | n_estimators | learning_rate | max_depth | dim reducer | n_component | scaler|
|---------|-------------|------------------|-------------|----------|----------|----------|
| 0.5787 | 152 | 0.3664 | 12 | PCA | 11 | PowerTransformer |

### Support Vector Machine

The hyperparameters to optimize are C (regularization) with range [0.1, 10), kernels have these options [linear, poly, rbf, sigmoid], and gamma with range [1, 10). The performance plot for the RSHB is shown below.

| 5-fold test CV mean accuracy | C | kernel | gamma | dim reducer | n_component | scaler|
|---------|-------------|------------------|-------------|----------|----------|----------|
|0.5589 | 7.1 | rbf | 6 | ICA | 17 | StandardScaler |

## Conclusion

The boxplot is shown below to compare the performance of the three models with different pipeline using Hyperband with random sampler.

![boxplot](./all_hbrs_perf_boxplot.png)

The figure above shows that the random forest has the best overall performance among the three models. It has highest average accuracy but there are some outliers. The gradient boosting has the second best performance and with the similar dispersion as the random forest., and the SVM has the worst performance. The SVM model has the highest dispersion and lowest average accuracy. The results on the test dataset also agree with this trend. The pipelines for random forest and gradient boosting both reply on PCA as dimension reducer and power transformation as scaler. Among six scalers, only power transformation and standard scaler are selected and SVM with standard scaler has much higher dispersion than the other two with power transformation. This would mean that many features in the dataset are not normally distributed and the power transformation is used to transform the data to be more Gaussian-like.

![perfplot](./perfplot.png)

The figure above shows the performance (convergence) of the three models with different pipelines in the tuning process. We can observe that all three models reach
plateau near 20 - 40 iterations before dropping, which means all models have explored regions with higher accuracy and there is a good balance between exploration and exploitation.
