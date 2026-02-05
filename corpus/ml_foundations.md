# Machine Learning Foundations

Reference material for core ML concepts used in the toolbox and in practice.

## Bias and Variance

**Bias** is the error due to the model being too simple (e.g. linear model when the true relationship is nonlinear). High bias often means underfitting: the model does not capture the main patterns in the data.

**Variance** is the error due to the model being too sensitive to the training set. High variance often means overfitting: the model memorizes noise and performs poorly on new data.

The **bias–variance tradeoff** says that reducing one usually increases the other. Regularization, more data, and model choice (e.g. simpler vs more flexible) all affect this tradeoff. The goal is to balance bias and variance so total error on unseen data is minimized.

## Overfitting and Underfitting

**Overfitting** occurs when the model fits the training data too closely, including noise, and generalizes poorly to new data. Signs: training accuracy much higher than validation/test accuracy, or validation loss increasing while training loss keeps decreasing. Remedies: more data, regularization (L1/L2), dropout, early stopping, or a simpler model.

**Underfitting** occurs when the model is too simple to capture the main structure in the data. Signs: both training and validation performance are poor. Remedies: more capacity (e.g. more layers or parameters), more features, or less regularization.

## Train, Validation, and Test Sets

The **training set** is used to fit the model (update weights or parameters).

The **validation set** is used to tune hyperparameters and to monitor generalization during training (e.g. for early stopping). It should not be used to fit the model.

The **test set** is used only at the end to report a final estimate of performance on unseen data. It should not influence any training or hyperparameter choices. Best practice is to fix the split (e.g. by random seed) so results are reproducible.

## Cross-Validation

**Cross-validation** (e.g. k-fold) uses multiple train/validation splits to get a more stable estimate of performance and to use data more efficiently. In k-fold CV, the data is split into k folds; each fold is used once as the validation set while the rest are used for training; the average (and standard deviation) of the validation metric is reported. Use cross-validation for model selection and hyperparameter tuning; keep a separate test set for final evaluation.

## Classification Metrics

For **classification**, common metrics include:

- **Accuracy**: fraction of correct predictions. Can be misleading when classes are imbalanced.
- **Precision**: among predicted positives, how many are actually positive. Important when false positives are costly.
- **Recall** (sensitivity): among actual positives, how many were predicted positive. Important when false negatives are costly.
- **F1 score**: harmonic mean of precision and recall; balances the two.
- **ROC-AUC**: area under the receiver operating characteristic curve; measures ranking quality across thresholds and is robust to class imbalance in many cases.

Choose metrics based on the business or scientific cost of false positives vs false negatives.

## Regression Metrics

For **regression**, common metrics include:

- **Mean squared error (MSE)**: average of squared errors; penalizes large errors heavily.
- **Root mean squared error (RMSE)**: square root of MSE; in the same units as the target.
- **Mean absolute error (MAE)**: average of absolute errors; less sensitive to outliers than MSE.
- **R² (coefficient of determination)**: proportion of variance in the target explained by the model; 1 is perfect, 0 means no better than predicting the mean.

## Ensemble Methods

**Ensembles** combine multiple models to improve robustness and often accuracy. Common approaches:

- **Voting**: average (regression) or majority vote (classification) over base models.
- **Bagging** (e.g. Random Forest): train many models on bootstrap samples of the data; reduce variance.
- **Boosting** (e.g. AdaBoost, gradient boosting): train models sequentially, each focusing on errors of the previous; reduce bias and variance.
- **Stacking**: use a meta-model to combine predictions of base models.

The toolbox provides evaluation, tuning, and ensemble utilities that work with these patterns.
