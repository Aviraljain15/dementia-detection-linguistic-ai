import os
import pandas as pd
import numpy as np
import pickle

from chat_parser import ChatParser
from feature_extractor import LinguisticFeatureExtractor
from baseline_model import run_baseline
from xgboost_model import run_xgboost
from bert_fusion_model import run_bert_fusion


# ================= PATH SETUP ================= #
BASE_PATH = os.path.join(os.path.dirname(__file__), "..", "Pitt")
TASKS = ["cookie", "fluency", "recall", "sentence"]


# ================= LOAD DATA ================= #
def load_samples():
    parser = ChatParser()
    samples = []

    for task in TASKS:
        print(f"\nLoading task: {task}")

        for label_name, label in [("Control", 0), ("Dementia", 1)]:
            folder = os.path.join(BASE_PATH, label_name, task)

            if not os.path.exists(folder):
                print(f"Missing folder: {folder}")
                continue

            for file in os.listdir(folder):
                if file.endswith(".cha"):
                    path = os.path.join(folder, file)

                    try:
                        sample = parser.parse_file(path, label)
                        samples.append(sample)
                    except Exception as e:
                        print(f" Error in {path}: {e}")
    return samples

# ================= MAIN PIPELINE ================= #
def main():
    print("\n===== DEMENTIA DETECTION PIPELINE =====\n")

    # 🔹 LOAD DATA
    samples = load_samples()
    print(f"\n Total samples loaded: {len(samples)}")

    if len(samples) == 0:
        print(" No data found. Check dataset path.")
        return
    # 🔹 FEATURE EXTRACTION
    print("\nExtracting linguistic features...")
    extractor = LinguisticFeatureExtractor()
    feature_rows = [extractor.extract(s) for s in samples]
    X = pd.DataFrame(feature_rows).fillna(0)
    y = np.array([s.label for s in samples])

    print(f" Feature matrix shape: {X.shape}")

    # ================= MODELS ================= #

    #  BASELINE
    print("\nRunning TF-IDF Baseline...")
    baseline_results = run_baseline(samples)

    #  XGBOOST
    print("\nRunning XGBoost Model...")
    xgb_model, xgb_preds, feature_names = run_xgboost(samples)

    #  BERT FUSION
    #print("\nRunning BERT Fusion Model...")
    #bert_model = run_bert_fusion(samples, X.values)

    # ================= SAVE OUTPUTS ================= #
    print("\nSaving outputs for visualization...")

    output_path = os.path.join(os.path.dirname(__file__), "outputs.pkl")
    with open(output_path, "wb") as f:
        pickle.dump({
            "samples": samples,
            "X": X,
            "y": y,
            "baseline": baseline_results,
            "xgb_preds": xgb_preds,
            "xgb_model": xgb_model,
            "feature_names": feature_names
        }, f)

    print(" outputs.pkl saved successfully")

    print("\n===== PIPELINE COMPLETE =====\n")


# ================= ENTRY ================= #
if __name__ == "__main__":
    main()