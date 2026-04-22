# Facial Recognition Fine-Tuning Pipeline


An end-to-end, highly modular PyTorch pipeline engineered to process, sanitize, fine-tune, and critically evaluate an `InceptionResnetV1` (FaceNet) architecture on deeply nested, low-quality image datasets.

Designed specifically for the **Jezt Technologies AI Developer Intern Evaluation**, this repository prioritizes strict data integrity, zero data leakage, and high-fidelity visual analytics.

---

##  Architectural Philosophy

This pipeline was built adhering to production-grade machine learning principles:
1. **Separation of Concerns:** Data extraction, augmentation, training, and evaluation are strictly compartmentalized into independent modules.
2. **Absolute Data Integrity:** A mathematical zero-leakage guarantee is enforced between the training and evaluation splits.
3. **Fault Tolerance:** Automated data sanitation sweeps prevent the PyTorch `DataLoader` from crashing due to corrupted or 0-byte files common in raw datasets.
4. **Robust Feature Extraction:** Dynamic data augmentation (Gaussian blur, color jitter) forces the model to learn structural embeddings rather than memorizing high-frequency noise inherent in low-quality inputs.

---

## 📂 Project Structure

```text
jezt_vision_pipeline/
│
├── Images/                     # Raw, unzipped dataset (Deeply nested subfolders)
├── dataset/                    # Auto-generated, sanitized, and flattened dataset
│   ├── train/                  # 80% split (Strictly locked from eval)
│   └── eval/                   # 20% split (Strictly locked from train)
│
├── models/                     # Saved PyTorch checkpoints (.pth)
│
├── 1_data_split.py             # Recursive extractor & zero-collision splitter
├── 2_dataset.py                # PyTorch DataLoader & Augmentation Engine
├── 3_train.py                  # Fine-tuning engine (Transfer Learning)
├── 4_evaluate.py               # Evaluator (Cosine Similarity, ROC, Visualizations)
├── 5_clean_data.py             # PIL-based corrupted file sanitizer
│
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation


#  Execution Pipeline

To reproduce the results, ensure you are in an isolated virtual environment and follow the steps below in strict order.

---

## ⚙️ Environment Setup

```bash
python -m venv jezt_env
```

### Activate Environment

```bash
# Linux / Mac
source jezt_env/bin/activate

# Windows
jezt_env\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 📌 Pipeline Execution (Strict Order)

### 1️⃣ Extract & Split Data

```bash
python data_split.py
```

**Expected Output:**

```
Zero leakage mathematically verified.
```

---

### 2️⃣ Sanitize Dataset

```bash
python clean_data.py
```

**Expected Output:**

```
Dataset is clean. Ready for PyTorch ingestion.
```

---

### 3️⃣ Establish Baseline Metrics

```bash
python evaluate.py
```

**Expected Output:**

* `Baseline_roc.png`
* `Baseline_distribution.png`

---

### 4️⃣ Execute Fine-Tuning

```bash
python train.py
```

**Expected Output:**

* Training across configured epochs
* Saved model:

```
models/finetuned_facenet.pth
```

---

### 5️⃣ Post-Tuning Re-Evaluation

```bash
python evaluate.py --weights models/finetuned_facenet.pth
```

---

## ✅ Final Outcome

This pipeline ensures:

* ✔️ Zero data leakage
* ✔️ Clean and validated dataset
* ✔️ Baseline performance benchmarking
* ✔️ Fine-tuned model training
* ✔️ Post-training evaluation

---

## 🧠 Notes

* Execute steps **in exact order** to maintain reproducibility
* Ensure dataset paths and configs are correctly set
* GPU acceleration is recommended for faster training


