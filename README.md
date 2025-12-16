# shift-planning-lightgbm
Machine learning pipeline to learn and predict assignment patterns in shift plans. Trains LightGBM on historical rosters, calibrates with isotonic regression, and for each shift outputs a probability for each employee to be assigned, plus ranked recommendations, optional fairness caps, and a visualization/reporting CLI.
