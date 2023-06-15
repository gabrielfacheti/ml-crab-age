<!DOCTYPE html>
<html>
		
<body>

<h1>Machine Learning: Crab Age Prediction</h1>
This notebook was developed using Python and its libraries as part of a Kaggle competition to predict the age of crabs based on physical attributes using regression techniques. 
The dataset provided for this competition was divided into a train set (74,051 rows and 10 columns) and a test set (49,368 rows and 9 columns). Both sets provide physical information about crabs, such as Length, Weight, Height, and more. The data is available <a href="https://www.kaggle.com/competitions/playground-series-s3e16">here</a>.<br>

<h2>Methodology</h2>

<h3>Exploratory Data Analysis and Feature Engineering</h3>
To gain insights into the dataset and improve its suitability for modeling, Exploratory Data Analysis (EDA) and Feature Engineering techniques were applied. EDA involved analyzing the dataset's characteristics, distributions, and relationships among variables. Feature Engineering aimed to create new features or transform existing ones to enhance the predictive power of the models.

<h3>Data Preprocessing</h3>
The following preprocessing steps were performed on the dataset before training the models:
<ul>
	<li>Min-Max Scaling: Numerical features were transformed using <code>MinMaxScaler</code>, which scales the features to a specified range (e.g., [0, 1]). This scaling technique ensures that all numerical features have a consistent range and aids in the convergence of regression algorithms.</li>
	<li>One-Hot Encoding: Categorical features were encoded using <code>OneHotEncoder</code> to convert them into numerical representations suitable for machine learning algorithms.</li>
</ul>
This step was achieved using <code>Pipeline</code> and <code>ColumnTransformer</code> to streamline the preprocessing pipeline.

<h3>Model Selection and Evaluation</h3>
To identify the best regression model for the task, cross-validation using <code>KFold</code> was employed. This technique splits the training dataset into K folds and iteratively trains and evaluates the models on different combinations of train and validation sets. Mean Absolute Error (MAE) was used as the performance metric to evaluate the regression models. MAE measures the average absolute difference between the predicted and actual values, providing an understanding of the model's average prediction accuracy.

<h3>Feature Importance Analysis</h3>
For the models that support feature importance analysis, the relative importance of each feature in predicting the crab's age was examined. This analysis provides insights into the significance of different physical attributes in estimating age accurately.

<h3>Hyperparameter Tuning</h3>
Two of the best-performing models were selected for further improvement through hyperparameter tuning. <code>RandomizedSearchCV</code> was utilized to explore different combinations of hyperparameters and identify the optimal configuration for each model. This process aims to enhance the models' predictive performance.

<h3>Ensemble Voting Regressor</h3>
The two tuned models were combined into a <code>VotingRegressor</code>, which aggregates the predictions from each model to make the final regression estimation. This ensemble approach leverages the strengths of each model and improves the overall accuracy of the age predictions.

<h2>Results</h2>
<p>The following image shows us the results of each model evaluated during the cross-validation.</p>
  
![image](https://github.com/gfacheti/ML-Crab-Age/assets/106284497/ee765f16-ab3a-48c2-a1d4-7d23cd71337b)

<p>Then I evaluated the feature importances of each model.</p>
  
![image](https://github.com/gfacheti/ML-Crab-Age/assets/106284497/7642daee-dbc3-4da0-ad99-cd66db355b12)

<p>As the best performers LGBM and CatBoost were selected for tuning. The following images show the results and their best parameters.</p>

![image](https://github.com/gfacheti/ML-Crab-Age/assets/106284497/5c202b2f-92fe-457f-92a2-ea3cef199b26)

    best_params = {'LGBM': {'n_estimators': 200, 'max_depth': 6, 'learning_rate': 0.15},
                   'CatBoost': {'learning_rate': 0.15, 'iterations': 300, 'depth': 6}}

<p>After the tuning, the models were ensembled in a VotingRegressor, which decreased the MAE.</p>
<ul>
    <li>RMSE: 2.0481</li>
    <li>MAE: 1.3471</li>
</ul>
<p>The following image shows the difference between predicted ages and the ones in the validation set.</p>

![image](https://github.com/gfacheti/ML-Crab-Age/assets/106284497/771fd1ad-a4d2-41e0-88c6-fb694996e388)

<p>Finally, the image below shows the distribution of predicted ages on the test set.</p>

![image](https://github.com/gfacheti/ML-Crab-Age/assets/106284497/cd2054ea-e186-45ce-a234-da1104785e51)

</body>
</html>
