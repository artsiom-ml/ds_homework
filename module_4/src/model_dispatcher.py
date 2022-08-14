from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost.sklearn import XGBClassifier

models = {
    "log_reg": LogisticRegression(
        penalty='l2',
        C=0.001,
        random_state=42
    ),
    "rf": RandomForestClassifier(
          n_estimators=400,
          max_depth=3,
          random_state=42),
}