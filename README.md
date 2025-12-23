# shift-planning-lightgbm
Machine learning pipeline to learn and predict assignment patterns in shift plans. Trains LightGBM on historical rosters, calibrates with isotonic regression, and for each shift outputs a probability for each employee to be assigned, plus ranked recommendations, optional fairness caps, and a visualization/reporting CLI.

## How to train and predict shift assignments

In `shift_assignment.py`, run the main method in order to start the training and prediction process. The input has to be a `planning_request_complete.json` and the model trains on `past_shift_plans` data. The target assignments are the ones in the current planning period.