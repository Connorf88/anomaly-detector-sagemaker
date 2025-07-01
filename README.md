# 🧠 Anomaly Detector on AWS SageMaker (Free Tier)

This repo ports a time-series anomaly detection pipeline from Azure ML to **AWS SageMaker**, using only **free-tier** resources. The model is built with `scikit-learn`’s Isolation Forest, and uses offline, reproducible Python scripts compatible with `nano`, `joblib`, and `boto3`.

---

## 🔍 Project Goals

- ✅ Zero-cost anomaly detection on AWS
- 🔧 Train and test using local scripts or SageMaker Notebooks
- 🧪 Batch inference with CSVs (no real-time endpoint required)
- 🎯 Transparent, nano-editable workflow with reproducible steps

---

## 📁 Project Structure

```bash
anomaly-detector-sagemaker/
├── data/                     # Input and output CSVs
│   └── timeseries.csv
│   └── predicted.csv
├── model/                    # Serialized ML model
│   └── isolation_forest.joblib
├── src/
│   ├── train.py              # Training script
│   └── inference.py          # Batch inference script
├── requirements.txt          # Dependencies
└── sagemaker_notebook.ipynb  # Optional: orchestration notebook
