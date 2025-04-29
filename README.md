# News Classification MLOps Pipeline

**Author**: Anuj Jaganath Said  
**Roll Number**: ME21B172

This repository implements a complete MLOps pipeline for classifying news articles using modern tools such as MLflow, DVC, Docker, Prometheus, and Grafana. It supports end-to-end automation from data processing to model deployment and monitoring.

---

## 📌 Features

- ✅ End-to-end pipeline: data extraction, training, fine-tuning, and deployment
- 🐳 Dockerized frontend/backend with FastAPI and HTML templates
- 📊 Real-time monitoring with Prometheus and Grafana
- 🔁 Feedback loop for retraining with user feedback
- 🎯 Metrics collection and visualization
- 🚀 ML model serving on REST endpoint

---

## 🖼️ Architecture Overview

![Image](https://github.com/user-attachments/assets/537d7f12-6c1e-43ff-9d79-e719396c51c0)

---

## 🔧 Setup Instructions

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd <repo-name>

```

### 2. Install Python Dependencies

Make sure you have Python 3.9+ installed.

```bash
pip install -r requirements.txt
```

### 3. Start the ML Pipeline

```bash
python3 model/run.py
```

This starts the MLflow server on **port 8080**, initializes DVC pipeline stages, and prepares the model for serving on **port 5002**.

---

### 4. Start the Web Application

Use Docker Compose to run the frontend/backend:

```bash
docker-compose up -d
```

- App accessible at: `http://localhost:8000`
- Prometheus metrics exposed at: `http://localhost:18001`

---

## 📈 Monitoring Setup

### 5. Start Prometheus

Make sure `Prometheus.yaml` is present in the project root. Then, run:

```bash
prometheus --config.file=Prometheus.yaml
```

Prmetheus runs on: `http://localhost:9090`

---

### 6. Start Grafana

Use the official Grafana Docker image:

```bash
docker run -d -p 3000:3000 grafana/grafana
```

Grafana Dashboard: `http://localhost:3000`

Use this to visualize metrics fetched by Prometheus.

---

## 🌐 Web Interface

- Homepage: [http://localhost:8000](http://localhost:8000/)
- Category filter: [http://localhost:8000/category/<category>](http://localhost:8000/category/%3Ccategory%3E)

---

## 🧪 API Endpoints

| Endpoint | Function |
| --- | --- |
| `GET /` | Home route listing news articles |
| `GET /category/{category}` | Filtered news by category |
| `POST /submit-feedback` | Submit corrections for retraining |

---

## ⚙️ Environment Variables

Set the following in your Docker `.env` file or as environment variables:

| Variable | Description |
| --- | --- |
| `DB_HOST` | PostgreSQL host |
| `DB_NAME` | Database name |
| `DB_USER` | Database user |
| `DB_PASSWORD` | Database password |
| `MODEL_SERVER_URL` | URL where the ML model is served |
| `PYTHONUNBUFFERED` | Set to `1` for real-time logging |

---

## 🛠️ Maintenance and Model Updates

To retrain or fine-tune the model:

1. Update `params.yaml`
2. Re-run the pipeline:

```bash
dvc repro
```

Or manually:

```bash
python3 model/run.py
```

Restart Docker containers if needed.

---

## 📦 Scaling Recommendations

- Increase Docker replicas for high traffic
- Use NVIDIA Docker if GPU is available
- For production use, migrate to Kubernetes

---

## 💬 Feedback Loop

- User-submitted corrections are stored at: `/data/news_feed.csv`
- These are picked up in the next training cycle automatically.
