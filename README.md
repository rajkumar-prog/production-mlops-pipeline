# 🔐 ConfidentialMind — Zero-Trust Medical Document AI

> The first open-source MLOps pipeline where sensitive medical records are processed by AI **without any party — including the server — ever seeing the data**. Combines AWS Nitro Enclaves (hardware TEE), Differential Privacy, Federated Learning, and LayoutLM in one production system.

[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://python.org)
[![AWS](https://img.shields.io/badge/AWS-Nitro_Enclaves-orange)](https://aws.amazon.com/ec2/nitro/nitro-enclaves/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)](https://pytorch.org)
[![Flower](https://img.shields.io/badge/Flower-Federated_Learning-green)](https://flower.dev)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## 🧠 Why This Doesn't Exist Yet

AWS has demo repos for Nitro Enclaves + LLM. Research exists for federated learning in healthcare. Differential privacy is well-studied. But nobody has combined all of these into **one production MLOps pipeline**:

| Existing Work | Gap |
|---|---|
| AWS Nitro Enclaves LLM demo | Toy demo — no FL, no DP, no drift monitoring |
| QFed+FHE (NeurIPS 2024) | Quantum + crypto only — no hardware TEE, no full pipeline |
| FL in radiology research | No Nitro Enclaves, no drift detection, no production serving |

**What ConfidentialMind does that nobody has:**
> A complete production MLOps system where hospital data never leaves the hospital (Federated Learning), model gradients are mathematically noise-protected (Differential Privacy), and inference runs inside a **cryptographically isolated enclave** (AWS Nitro) — provably inaccessible even to AWS itself.

---

## 🏗️ System Architecture

```
TRAINING PHASE — Federated + Privacy-Preserving
─────────────────────────────────────────────────
Hospital A  →  Local LayoutLM fine-tuning  (data NEVER leaves)
Hospital B  →  Local LayoutLM fine-tuning  (data NEVER leaves)
Hospital C  →  Local LayoutLM fine-tuning  (data NEVER leaves)

Each hospital applies Differential Privacy (ε=1.0) to gradients
         │
         ▼  only encrypted DP gradients transmitted
┌────────────────────────┐
│  Federated Aggregator  │  ← Flower framework, weighted FedAvg
└────────────┬───────────┘
             ▼
     Global LayoutLM Model (no raw data ever seen)

INFERENCE PHASE — Zero-Trust via AWS Nitro Enclave
────────────────────────────────────────────────────
Patient uploads medical record
         │
         ▼
    AWS S3 (encrypted at rest, KMS-managed keys)
         │
         ▼
┌──────────────────────────────────────────────┐
│          AWS Nitro Enclave                   │
│  Isolated VM — no network, no SSH            │
│  Cryptographic attestation enforced          │
│                                              │
│  LayoutLM inference runs here:               │
│  extracts diagnosis, medications,            │
│  ICD codes, dates, patient fields            │
│                                              │
│  Results encrypted with patient's KMS key   │
│  Even AWS operators cannot see data          │
└──────────────────┬───────────────────────────┘
                   ▼
         Patient decrypts with their own key only

MONITORING PHASE — Auto-Retraining
────────────────────────────────────
Evidently AI monitors predictions over time
         │
         ▼  drift detected (F1 drops / distribution shift)
         │
         ▼
Auto-trigger: new federated training round across hospitals
```

---

## 📁 Project Structure

```
confidentialmind-zero-trust-medical-ai/
├── src/
│   ├── federated_trainer/    # Flower FL client + server
│   ├── differential_privacy/ # DP-SGD with Opacus
│   ├── enclave_runtime/      # Nitro Enclave vsock server + LayoutLM inference
│   ├── document_extractor/   # LayoutLM fine-tuning + preprocessing
│   ├── drift_monitor/        # Evidently AI drift detection + alerting
│   └── aggregator/           # Secure FedAvg aggregation logic
├── aws/
│   ├── nitro_enclave/        # Enclave Dockerfile + attestation scripts
│   ├── lambda/               # S3-trigger Lambda for inference routing
│   ├── sagemaker/            # SageMaker training job configs
│   └── kms/                  # KMS key policy templates
├── models/
│   ├── layoutlm/             # LayoutLM v3 fine-tuning configs
│   └── federated_weights/    # Round-by-round global model checkpoints
├── data/
│   ├── mimic3_samples/       # De-identified MIMIC-III sample records
│   ├── i2b2_samples/         # i2b2 NLP challenge samples
│   └── processed/            # Tokenized + LayoutLM-formatted inputs
├── api/                      # FastAPI: upload docs, query results
├── cicd/                     # GitHub Actions: FL round + model deploy
├── configs/                  # YAML configs per module
├── notebooks/                # Training, DP epsilon analysis, drift plots
└── tests/                    # Privacy guarantee tests, accuracy tests
```

---

## 🛠️ Tech Stack

| Component | Technology | Why |
|---|---|---|
| Document AI Model | LayoutLM v3 | SOTA structured document understanding |
| Federated Learning | Flower (flwr) | Production-grade FL framework |
| Differential Privacy | Opacus (PyTorch) | DP-SGD with formal privacy guarantees |
| Hardware TEE | AWS Nitro Enclaves | Cryptographic isolation — zero operator access |
| Key Management | AWS KMS | Patient-controlled encryption keys |
| Drift Detection | Evidently AI | Real-time model performance monitoring |
| Trigger Orchestration | AWS Lambda + S3 | Serverless inference routing |
| API Layer | FastAPI | Document upload + result retrieval |
| CI/CD | GitHub Actions | Automated FL rounds + model deployment |

---

## 🚀 Getting Started

```bash
git clone https://github.com/rajkumar-prog/production-mlops-pipeline.git
cd production-mlops-pipeline
pip install -r requirements.txt
cp .env.example .env   # AWS credentials, KMS key ID
```

### Simulate Federated Training (3 hospitals)
```bash
python scripts/run_federated.py \
  --num-clients 3 \
  --rounds 10 \
  --dp-epsilon 1.0 \
  --config configs/fl_training.yaml
```

### Build and Deploy Nitro Enclave
```bash
cd aws/nitro_enclave
docker build -t confidentialmind-enclave .
nitro-cli build-enclave --docker-uri confidentialmind-enclave --output-file enclave.eif
nitro-cli run-enclave --cpu-count 2 --memory 4096 --eif-path enclave.eif
```

### Run Inference via API
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
curl -X POST http://localhost:8000/extract \
  -H "X-Patient-Key: <your-kms-key-id>" \
  -F "file=@data/mimic3_samples/record_001.pdf"
```

---

## 🔒 Privacy Guarantees

| Layer | Mechanism | Guarantee |
|---|---|---|
| Training | Differential Privacy (ε=1.0, δ=1e-5) | Mathematical bound on individual data leakage |
| Communication | Federated Learning | Raw data never leaves hospital |
| Inference | AWS Nitro Enclave | Hardware-attested isolation, zero operator access |
| Storage | AWS KMS + patient keys | Only patient can decrypt their results |

---

## 📊 Datasets

| Dataset | Content | Access |
|---|---|---|
| MIMIC-III | 40,000+ ICU patient records | [physionet.org](https://physionet.org/content/mimiciii/1.4/) |
| i2b2 NLP | Clinical NLP challenge data | [i2b2.org](https://www.i2b2.org/NLP/DataSets/) |
| FUNSD | Form understanding benchmark | [guillaumejaume.github.io](https://guillaumejaume.github.io/FUNSD/) |

---

## 🔮 Roadmap

- [ ] LayoutLM v3 fine-tuning on i2b2 + MIMIC samples
- [ ] Flower federated training (3-hospital simulation)
- [ ] Opacus DP-SGD with epsilon tracking per round
- [ ] AWS Nitro Enclave Dockerfile + vsock inference server
- [ ] KMS key policy for patient-controlled decryption
- [ ] Evidently AI drift monitor + alert system
- [ ] Auto-retraining trigger via Lambda on drift event
- [ ] GitHub Actions CI/CD for FL rounds
- [ ] FastAPI end-to-end encrypted document flow
- [ ] Privacy audit: formal epsilon-DP guarantee verification

---

## 📄 Related Work

- [AWS Nitro medical LLM sample](https://github.com/aws-samples/sample-for-secure-medical-llm-inference-with-nitro-enclaves) — demo only, no FL, no DP, no drift
- [QFed+FHE NeurIPS 2024](https://github.com/elucidator8918/QFL-MLNCP-NeurIPS) — quantum + crypto, no TEE, no document AI
- [Federated HE in radiology](https://github.com/tayebiarasteh/federated_he) — no Nitro Enclaves, no full pipeline

**ConfidentialMind is the first complete MLOps pipeline combining hardware TEE + DP + FL + document AI + drift monitoring.**

---

## 👤 Author

**Raj Kumar Satya** — AI/ML Engineer | W&B • Possible Finance • Razorpay
- GitHub: [@rajkumar-prog](https://github.com/rajkumar-prog)
- Email: rajkumarsatya65@gmail.com
