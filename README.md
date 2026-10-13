# Predicting Personal Loan Acceptance Using Machine Learning

## Project Overview

This project predicts whether a bank customer is likely to accept a
personal loan offer using customer demographic, financial, and banking
information. It uses the **Bank Personal Loan Modelling** dataset and
compares five machine learning approaches:

-   Perceptron
-   Logistic Regression
-   Support Vector Machine (SVM)
-   Multi-Layer Perceptron (MLP)
-   K-Means clustering

The project follows the core problem, dataset, and five algorithms
described in the reference paper, *Predicting Acceptance of Personal
Loans Using Machine Learning Algorithms*. It extends the comparison
through hyperparameter tuning and evaluation metrics suited to an
imbalanced binary classification problem.

## Team Details

  Name                   SRN
  ---------------------- ---------------
  M S NIKHIL CHOWDHARY   PES2UG24CS253
  MOHITH KUMAR BATTU     PES2UG24CS283

## Objectives

-   Predict whether a customer will accept a personal loan offer.
-   Compare five machine learning algorithms on the same dataset and
    train/test split.
-   Tune model hyperparameters using cross-validation.
-   Evaluate performance with metrics beyond accuracy, especially
    F1-score and ROC-AUC.
-   Visualize and compare the performance of the trained models.

## Dataset

**Dataset:** Bank Personal Loan Modelling

The dataset contains **5,000 customer records** and 14 columns,
including the customer ID, 12 input features, and the target column
`Personal Loan`.

### Input features

The `ID` column is excluded from model training because it is an
identifier rather than a predictive feature.

  -----------------------------------------------------------------------
  Feature                             Description
  ----------------------------------- -----------------------------------
  Age                                 Customer's age

  Experience                          Years of professional experience

  Income                              Annual income

  ZIP Code                            Customer's home ZIP code

  Family                              Family size

  CCAvg                               Average monthly credit-card
                                      spending

  Education                           Education level

  Mortgage                            Value of the customer's house
                                      mortgage, if any

  Securities Account                  Whether the customer has a
                                      securities account with the bank

  CD Account                          Whether the customer has a
                                      certificate of deposit account

  Online                              Whether the customer uses the
                                      bank's online banking facilities

  CreditCard                          Whether the customer has a credit
                                      card issued by the bank
  -----------------------------------------------------------------------

**Target:** `Personal Loan`

-   `0` --- Customer did not accept the personal loan.
-   `1` --- Customer accepted the personal loan.

The target is imbalanced: approximately 90.4% of records belong to class
`0` and 9.6% to class `1`. Therefore, accuracy alone is not sufficient
to assess the models.

Place `Bank_Personal_Loan_Modelling.csv` in the location expected by the
notebook, or update the CSV path in the data-loading cell.

## Methodology

### 1. Exploratory Data Analysis

The notebook explores the dataset using:

-   Dataset shape, column information, and descriptive statistics
-   Missing-value and duplicate checks
-   Target-class distribution
-   Feature correlation heatmap
-   Visual comparisons of selected features against loan acceptance

### 2. Preprocessing and data split

-   The `ID` and `Personal Loan` columns are excluded from the input
    features.
-   The target is stored separately.
-   Data is split into **80% training data and 20% test data**.
-   Stratification preserves target-class proportions across the split.
-   `random_state=42` makes the split reproducible.
-   Standard scaling is used for the models that require it. For
    Perceptron, Logistic Regression, SVM, and MLP, scaling is included
    in a scikit-learn `Pipeline` so that each cross-validation fold fits
    its own scaler. K-Means uses scaling fitted on the training data,
    and its cross-validation procedure scales within each fold.

### 3. Model training and hyperparameter tuning

The notebook uses `GridSearchCV` with **5-fold cross-validation** and
**F1-score** as the tuning metric. F1 is used because the positive class
is the minority class and a model can obtain high accuracy while missing
loan acceptances.

  -----------------------------------------------------------------------
  Algorithm                           Approach
  ----------------------------------- -----------------------------------
  Perceptron                          Tuned with different
                                      regularization, learning-rate, and
                                      iteration settings

  Logistic Regression                 Tuned with different regularization
                                      strengths, penalties, and
                                      compatible solvers

  SVM                                 Compares linear, RBF, and
                                      polynomial kernels and their
                                      parameters

  MLP                                 Tunes hidden-layer architecture,
                                      activation, regularization, and
                                      learning-rate settings

  K-Means                             Searches clustering parameters and
                                      maps the cluster with the higher
                                      loan-acceptance rate to class `1`,
                                      based on training data
  -----------------------------------------------------------------------

**K-Means note:** K-Means is an unsupervised clustering algorithm, not a
conventional supervised classifier. Since its cluster labels do not
inherently correspond to loan acceptance or rejection, the
implementation maps the cluster with the higher acceptance rate in the
training data to class `1`. The mapping is determined separately within
each cross-validation fold. For ROC-AUC, a continuous score is derived
from the distances to the two cluster centroids.

### 4. Evaluation

Each model is evaluated on the held-out test set using:

-   **Accuracy:** Fraction of all predictions that are correct.
-   **Precision:** Fraction of predicted loan acceptances that are
    actual acceptances.
-   **Recall:** Fraction of actual loan acceptances that are identified.
-   **F1-score:** Harmonic mean of precision and recall.
-   **ROC-AUC:** Measures how well the model ranks positive cases above
    negative cases across classification thresholds.

The notebook also generates classification reports, confusion matrices,
accuracy and F1 comparison plots, an ROC curve comparison, an AUC
comparison, and a summary of the best hyperparameters.

## Results

The following results are from the final notebook version discussed for
this project.

  Model                     Accuracy    Precision       Recall     F1-score      ROC-AUC
  --------------------- ------------ ------------ ------------ ------------ ------------
  Perceptron                  0.9290       0.6238       0.6562       0.6396       0.9400
  Logistic Regression         0.9520       0.8077       0.6562       0.7241       0.9633
  SVM                         0.9840       0.9545       0.8750       0.9130       0.9924
  MLP                     **0.9870**   **0.9560**   **0.9062**   **0.9305**   **0.9930**
  K-Means                     0.8300       0.3405       0.8229       0.4817       0.8888

### Discussion

-   **MLP** achieves the highest test accuracy, F1-score, and ROC-AUC in
    this experiment.
-   **SVM** is a close second and also provides strong precision,
    recall, and F1-score.
-   **Logistic Regression** performs better than the Perceptron on
    F1-score and ROC-AUC.
-   **K-Means** identifies many positive cases, resulting in relatively
    high recall, but its low precision indicates that it also predicts
    many customers as likely to accept a loan when they do not.
-   The results show why accuracy should be considered alongside
    precision, recall, F1-score, and ROC-AUC for an imbalanced dataset.

These figures are the results of this project's tuned models and test
split. They should not be presented as exact reproductions of the
reference paper's original configurations or reported results.

## Reference Paper

AJ Rossman and Conor Casey, **"Predicting Acceptance of Personal Loans
Using Machine Learning Algorithms."**

The reference work studies the same personal-loan acceptance problem and
the five algorithms used here. This project extends the experimental
comparison by performing systematic hyperparameter tuning and reporting
several evaluation metrics.

## Requirements

The notebook uses Python and the following main libraries:

-   pandas
-   NumPy
-   Matplotlib
-   Seaborn
-   scikit-learn
-   Jupyter Notebook or Google Colab

## How to Run

### Option 1: Google Colab

1.  Upload `Loan_Acceptance_Prediction.ipynb` to Google Colab.
2.  Upload `Bank_Personal_Loan_Modelling.csv` to the Colab session, or
    connect the notebook to the location where the dataset is stored.
3.  Check the CSV path in the data-loading cell.
4.  Run the notebook from top to bottom.

### Option 2: Run locally

1.  Install Python and the required packages:

    ``` bash
    pip install -r requirements.txt
    ```

2.  Place the notebook and CSV in the expected project location.

3.  Start Jupyter:

    ``` bash
    jupyter notebook
    ```

4.  Open `Loan_Acceptance_Prediction.ipynb` and run all cells from top
    to bottom.

**Note:** Grid search can take some time because multiple hyperparameter
combinations are evaluated using five-fold cross-validation.

## Reproducibility

-   The train/test split uses `random_state=42`.
-   The notebook uses a fixed evaluation workflow and cross-validation
    for hyperparameter selection.
-   Run the notebook from the beginning to reproduce the results in the
    same environment. Library versions, numerical behavior, and platform
    differences may cause small variations.

## Limitations

-   The dataset is imbalanced, with relatively few customers accepting a
    personal loan.
-   Results are based on one train/test split; performance may vary on
    other samples or real-world data.
-   ZIP Code is included as a feature, but its usefulness and potential
    geographic bias should be considered before real deployment.
-   K-Means is designed for clustering, so converting its clusters into
    loan predictions is an experimental comparison rather than a
    like-for-like supervised classification method.
-   The models demonstrate predictive performance on this dataset and
    should not be used to make real lending decisions without additional
    validation, fairness analysis, governance, and domain review.

## Future Work

Potential extensions include:

-   Evaluating results across multiple random train/test splits.
-   Comparing class-weighting and resampling methods for the minority
    class.
-   Calibrating predicted probabilities and selecting decision
    thresholds based on business costs.
-   Examining feature importance and model explainability.
-   Testing robustness and fairness across customer groups.
-   Validating performance on newer or external data.
