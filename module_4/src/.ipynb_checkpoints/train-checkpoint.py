import os
import argparse
import joblib
import pandas as pd
from sklearn.metrics import roc_auc_score
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import logging
from configs import config
import model_dispatcher
import preprocessing



def run(model):
    df = pd.read_csv(config.TRAINING_FILE)
    x_train, y_train, x_valid, y_valid = preprocessing.preprocess(df)
    try:
        clf = model_dispatcher.models[model]
    except KeyError:
        logging.error("Model not found")
        return
    
    clf.fit(x_train, y_train)

    preds = clf.predict_proba(x_valid) [:,1]

    roc_auc = roc_auc_score(y_valid, preds)
    print(f"Model={model}, roc_auc_score={roc_auc:.3f}")
    
    joblib.dump(
        clf,
        os.path.join(config.MODEL_OUTPUT, f"{model}.bin")
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', required=True, type=str,)

    args = parser.parse_args()

    run(
        model=args.model
    )