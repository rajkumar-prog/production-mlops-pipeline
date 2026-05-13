# 📦 Production MLOps Pipeline — Intelligent Document AI

> End-to-end MLOps platform for intelligent document processing — powered by AWS SageMaker, LayoutLM, FastAPI, and automated CI/CD with model drift detection.
> Built to demonstrate Amazon AWS AI-style production ML infrastructure.

[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://python.org)
[![AWS](https://img.shields.io/badge/AWS-SageMaker%20%7C%20Lambda%20%7C%20S3-orange)](https://aws.amazon.com)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-Containerized-blue)](https://docker.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## 🎯 Project Overview

This project implements a fully automated, cloud-native MLOps pipeline that ingests documents (invoices, contracts, financial forms), extracts structured data using a fine-tuned LayoutLM model, applies LLM-based reasoning for summarization, monitors model performance in production, and automatically triggers retraining when accuracy degrades — all on AWS.

---

## 🏗️ Architecture

```
Document Upload (PDF/Image)
         │
         ▼
    AWS S3 Bucket
         │  S3 Event
         ▼
   AWS Lambda Trigger
         │
         ▼
┌────────────────────┐
│  Preprocessing     │  ← PDF parsing, OCR (Tesseract), image normalization
└────────┬───────────┘
         │
         ▼
┌────────────────────┐
│  SageMaker         │  ← LayoutLM / Donut inference endpoint
│  Inference         │     extracts: dates, amounts, entities, fields
└────────┬───────────┘
         │
         ▼
┌────────────────────┐
│  LLM Reasoning     │  ← OpenAI / Claude API → summarization + validation
│  Layer             │
└────────┬───────────┘
         │
         ▼
┌────────────────────┐
│  PostgreSQL +      │  ← Store extracted fields + confidence scores
│  Redis Cache       │
└────────┬───────────┘
         │
         ▼
┌────────────────────┐
│  Monitoring &      │  ← Track accuracy, latency, data drift (Evidently AI)
│  Drift Detection   │     Auto-trigger SageMaker retraining on drift
└────────┬───────────┘
         │
         ▼
┌────────────────────┐
│  FastAPI Dashboard │  ← REST API + web UI for results
└────────────────────┘
         │
   CI/CD Pipeline
   (GitHub Actions)
   Auto-deploys on push
```

---

## 📁 Project Structure

```
production-mlops-pipeline/
├── src/
│   ├── ingestion/          # S3 upload handlers, PDF parsing, OCR
│   ├── extraction/         # LayoutLM / Donut model inference
│   ├── inference/          # LLM reasoning layer (OpenAI API)
│   └── monitoring/         # Drift detection, alerting, metrics
├── aws/
│   ├── lambda/             # Lambda function handlers
│   └── sagemaker/          # SageMaker training & deployment scripts
├── api/                    # FastAPI application
├── cicd/                   # GitHub Actions workflows
├── configs/                # Pipeline & model config YAMLs
├── data/
│   ├── raw/                # Sample documents
│   └── processed/          # Parsed & tokenized data
├── models/                 # Model artifacts & fine-tuning configs
├── notebooks/              # Training, evaluation, EDA notebooks
└── tests/                  # Unit, integration, and load tests
```

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Document AI Model | LayoutLM v3 / Donut |
| LLM Reasoning | OpenAI GPT-4o / Claude API |
| Cloud Platform | AWS (SageMaker, Lambda, S3, EC2) |
| Serving API | FastAPI |
| Database | PostgreSQL + Redis |
| Containerization | Docker |
| CI/CD | GitHub Actions |
| Drift Detection | Evidently AI |
| Experiment Tracking | MLflow / W&B |
| OCR | Tesseract / AWS Textract |

---

## 🚀 Getting Started

### Installation
```bash
git clone https://github.com/rajkumar-prog/production-mlops-pipeline.git
cd production-mlops-pipeline
pip install -r requirements.txt
```

### Configure AWS
```bash
aws configure  # set your AWS credentials
cp .env.example .env  # fill in API keys
```

### Run Locally (without AWS)
```bash
docker-compose up --build
# API available at http://localhost:8000
```

### Process a Document
```bash
curl -X POST http://localhost:8000/extract \
  -F "file=@data/raw/invoice_sample.pdf"
```

### Deploy to AWS
```bash
bash scripts/deploy_sagemaker.sh
bash scripts/deploy_lambda.sh
```

---

## 📊 Performance

| Metric | Value |
|---|---|
| Document Processing Time | TBD |
| Field Extraction Accuracy | TBD |
| API Latency (p95) | TBD |
| Monthly Scale | TBD |

---

## 🔮 Roadmap

- [ ] PDF ingestion pipeline with OCR
- [ ] LayoutLM fine-tuning on invoice dataset
- [ ] SageMaker training & deployment scripts
- [ ] Lambda trigger on S3 upload
- [ ] FastAPI REST endpoint with auth
- [ ] PostgreSQL + Redis integration
- [ ] Drift detection with Evidently AI
- [ ] Auto-retraining trigger on drift
- [ ] GitHub Actions CI/CD pipeline
- [ ] Web dashboard for document results

---

## 👤 Author

**Raj Kumar Satya**
- GitHub: [@rajkumar-prog](https://github.com/rajkumar-prog)
- LinkedIn: [Raj Kumar Satya](https://linkedin.com/in/rajkumarsatya)
- Email: rajkumarsatya65@gmail.com

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
