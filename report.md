# Hyperparameter optimization report

## Introduction

This report is a summary of the pipeline optimization process for the models used to predict the white wine quality with `winequality-white.csv` dataset. This pipeline optimization is largely based on the previous assignment on hyperparameter optimization with additions on the preprocessing tunning. Since there are only seven different quality levels (i.e. 3-9), I treat this problem as a classification problem. The pipeline consists of three components: a scaler, a dimension reducer, a ML model. The goal is to find the best preprocessing methods and hyperparameters for the pipeline to maximize the accuracy of the predictions. Three ML models are used in this report: Random Forest, Gradient Boosting, and Support Vector Machine. The scalers are `RobustScaler`, `PowerTransformer`, `Normalizer`, `StandardScaler`, `MinMaxScaler`, and `StandardScaler`. The dimension reducers are `PCA`, `FastICA`.

## Dataset description

The white wine dataset contains 4898 observations with 12 columns in each observation. There is no missing data in the dataset. The target variable is `quality`, which is an integer between 3 and 9. The rest of the columns are features of the wine. I consider all feature columns as continuous numerical variables. Due to the imbalanced dataset, I use SMOTE to balance the dataset. I have  The dataset is divided into 70% training and 30% testing.

Before SMOTE, the distribution of the quality levels is shown below.

![quality distribution before SMOTE](./before_bal.png)

After SMOTE, the distribution of the quality levels is shown below.

![quality distribution after SMOTE](./after_bal.png)

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

The hyperparameters to optimize are number of estimators with range [10, 500), max tree depth [1, 30), max number of features [1, 30), min samples split with range [2, 50) and min samples leaf range [1, 50). The best hyperparameters are shown in the table below.

| Category |5-fold test CV mean accuracy | n_estimators | max_depth | max_features | min_samples_split | min_samples_leaf | dim reducer | n_component | scaler|
|---------|-------------|------------------|-------------|-------------|-------------|-------------|-------------|----------|----------|
| Tuned pipeline |  0.6754 | 240 | 21 | 3 | 3 | 1 | PCA | 18 | MinMaxScaler |
| Default model only | 0.6878 | 100 | None | sqrt | None | 0.0 | NA | NA | NA |

### Gradient Boosting

The hyperparameters to optimize are number of estimators with range [10, 500), learning rate [0.01, 0.5) and max depth [1, 30).

The best hyperparameters are shown in the table below.

| Category | 5-fold test CV mean accuracy | n_estimators | learning_rate | max_depth | min_samples_split | min_samples_leaf | dim reducer | n_component | scaler|
|---------|---------|-------------|------------------|-------------|----------|----------|----------|----------|----------|
| Tuned pipeline | 0.6559 | 384 | 0.1387 | 23 | 11 | 6 | ICA | 4 | MinMaxScaler |
| Default model only | 0.6253 | 100 | 0.1 | 3 | 2 | 1 | NA | NA | NA |

### Support Vector Machine

The hyperparameters to optimize are C (regularization) with range [0.1, 10), kernels have these options [linear, poly, rbf, sigmoid], and gamma with range [1, 10). The performance plot for the RSHB is shown below.

| Category | 5-fold test CV mean accuracy | C | kernel | gamma | dim reducer | n_component | scaler|
|---------|---------|-------------|------------------|-------------|----------|----------|----------|
| Tuned pipeline | 0.6576 | 7.5 | rbf | 6 | ICA | 6 | MinMaxScaler |
| Default model only | 0.3048 | 1 | rbf | 'scale' | NA | NA | NA |


## Conclusion

The boxplot is shown below to compare the performance of the three models with different pipeline using Hyperband with random sampler.

![boxplot](./all_hbrs_perf_boxplot.png)

The figure above shows that the gradient boosting has the best overall performance among the three models. It has highest average accuracy but there are some outliers. The random forest has the second best performance. The SVM has the worst performance. The SVM model has the highest dispersion and lowest average accuracy. The results on the test dataset also agree with this trend. 

![perfplot](./perfplot.png)

The figure above shows the performance (convergence) of the three models with different pipelines in the tuning process. We can observe that gradient boosting has a very consistent good performance while random forest and SVC have a similar oscillating accuracy.
From three tables above, we can observe that the tuned random forest pipeline's performance is slightly worse than that of the default model-only pipeline. While the performance for
the tuned gradient boosting and SVM pipelines are much better than that of the default model-only pipelines. This indicates that the hyperparameter optimization process is effective for gradient boosting and SVM models but not for the random forest model. 

The RF classifier pipeline were actually tuned for more than 200 iterations,
twice as long as
the other models, but the tuned RF pipeline still performed worse than the default model-only pipeline. From the ablation study for my wildcard project, I found that the default RF classifier had a relatively good performance for many datasets, which means
it is very generalizable. This might be the reason why the tuned RF pipeline performed worse than the default model-only pipeline. The hyperparameter optimization process might not be effective for the RF model. The reason is because the search iterations might not high enough to find the globally best hyperparameters. The plateau in the performance plots show that at least the HPO process reached some locally optimal parameters.

To improve the performance of the random forest model, I would increase the search iterations further and search space for the hyperparameters. I would also use a different hyperparameter optimization algorithm such as Bayesian optimization (BOHB).