---
title: Serenity Backend
emoji: 🪐
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# Serenity Chatbot: Low-Latency Empathetic Mental Health Support System (Production Release)

Serenity is an empathetic mental health support chatbot system designed to deliver immediate, clinically guided, and safe conversational assistance to users. This repository houses the production-ready **Asynchronous FastAPI Backend API**, containerized with Docker, monitored via OpenTelemetry and Axiom, and deployed automatically through a GitHub Actions CI/CD pipeline.

---

## 📺 Project Demo Video & Live Interface

Below is the end-to-end system demonstration showing the chatbot frontend interacting in real-time, handling diverse intents (greetings, mental health queries, out-of-scope requests), receiving thumbs up/down user feedback, and visualizing operational telemetry in Axiom.

<div align="center">
  <h3>🎬 System Walkthrough Demo</h3>
  <video width="720" height="405" controls>
    <source src="assets/mental-health-chatbot-video.mp4" type="video/mp4">
    Your browser does not support the video tag.
  </video>
</div>

---

## 🎨 User Interface & Frontend Design

The Serenity client is a modern, responsive web interface built to foster a calming, premium user experience.

<div align="center">
  <img src="assets/Serenity_Frontend.png" alt="Serenity Frontend Chat Interface" width="720"/>
  <p><i>Figure 1: Premium frontend interface demonstrating responsive message bubbles, typing indicators, and user feedback buttons.</i></p>
</div>

---

## 🚀 Low-Latency Asynchronous Architecture (30x Speedup)

Initially, the backend codebase ran a synchronous thread-blocking pipeline. In that configuration, network API calls (Qdrant vector search, Groq completions) and CPU-heavy classifier inferences ran sequentially, locking Uvicorn's execution thread and resulting in high latencies (30–60+ seconds per message).

To resolve this bottleneck, we refactored the entire stack to be **fully asynchronous (Async/Await)**:
1. **Async Network Clients**: Refactored the database and LLM pipelines to use `AsyncQdrantClient` and `AsyncGroq`, ensuring that external network wait times do not lock the server threads.
2. **Concurrent CPU Tasks**: Leveraged `fastapi.concurrency.run_in_threadpool` wrapped inside `asyncio.gather` to execute the local scikit-learn language detector and fine-tuned PyTorch emotion classifier concurrently in non-blocking worker threads.
3. **Async Endpoints**: Promoted `/chat` and `/feedback` API routes to `async def` and wrapped SQLite database transactions inside asynchronous worker thread pools.

**Result**: Average message processing latency dropped from 50 seconds to **`1.65 seconds`**—achieving a **`30x performance acceleration`** under live conditions.

---

## 🛠️ API Endpoints & Contract

The backend exposes three core API endpoints:

### 1. `POST /chat`
* **Purpose**: Accepts a user message, executes the safety-guarded NLP RAG pipeline, and returns the response.
* **Request Shape (`ChatRequest`)**:
  ```json
  {
    "message": "I feel very down and anxious today."
  }
  ```
* **Response Shape (`ChatResponse`)**:
  ```json
  {
    "response": "It's completely okay to feel down and anxious sometimes. I am here to listen...",
    "answer": "It's completely okay to feel down and anxious sometimes. I am here to listen...",
    "session_id": "41.42.182.186",
    "emotion": "fear",
    "emotion_conf": 0.9414,
    "language": "en",
    "intent": "asking_mental_health_question",
    "crisis_flag": false
  }
  ```

### 2. `POST /feedback`
* **Purpose**: Logs user ratings (thumbs up/down) to audit model generations.
* **Request Shape (`FeedbackRequest`)**:
  ```json
  {
    "vote": "up",
    "user_message": "I feel very down and anxious today.",
    "bot_response": "It's completely okay to feel down and anxious sometimes..."
  }
  ```
* **Response**: `{"status": "success", "message": "Feedback saved successfully"}`

### 3. `GET /health`
* **Purpose**: Returns API service health status and the count of active memory sessions.
* **Response**: `{"status": "ok", "active_sessions": 1}`

---

## 🧠 NLP Pipeline Flow & Architecture

Serenity's pipeline handles messages using a layered, safety-first workflow:

```
               [ User Message ]
                      │
                      ▼
         ┌───────────────────────────┐
         │ 1. Language Detection     │  <-- joblib Sklearn Language Classifier
         └─────────────┬─────────────┘
                       ▼
         ┌───────────────────────────┐
         │ 2. Emotion Classification │  <-- PyTorch Hugging Face Classifier (DistilBERT)
         └─────────────┬─────────────┘
                       ▼
         ┌───────────────────────────┐
         │ 3. Safety Check / Intent  │  <-- Hardcoded triggers + Groq LLM Classifier
         └──────┬──────────────┬─────┘
                │              │
                │ (Crisis)     │ (General / Mental Health Query)
                ▼              ▼
      [Inject Hotline Banner]  [Qdrant RAG Retrieval]
                │              │
                │              ▼
                │              [Emotion Reranking & Quality Gates]
                │              │
                │              ▼
                │              [Intelligence Heuristic / Decision Gate]
                │              │
                ▼              ▼
         ┌───────────────────────────┐
         │ 4. Prompt Engineering     │  <-- Combines context, history, and emotion tones
         └─────────────┬─────────────┘
                       ▼
         ┌───────────────────────────┐
         │ 5. Response Generation    │  <-- Groq Llama 3 70B LLM Completion
         └───────────────────────────┘
```

1. **Language Locking**: Evaluates if the query is in English or Arabic to enforce native Egyptian Arabic output when needed.
2. **Emotion Profiling**: Local PyTorch sequence classification evaluates emotional state.
3. **Intent and Safety Guard**: Checks for crisis keywords (suicidal ideation). If flagged, immediately routes to safety protocols, overriding standard conversational flow. Otherwise, uses few-shot LLM prompts to classify the intent (`greeting`, `goodbye`, `mental_health_question`, `out_of_scope`).
4. **Context Retrieval (RAG)**: Embeds queries locally via `all-MiniLM-L6-v2` and searches Qdrant Cloud. Applies an **Emotion Reranking Heuristic** that boosts similarity scores for documents with topics matching the user's emotional state or exhibiting empathetic attributes.
5. **Intelligence Heuristic**: Reviews the top search scores. If retrieval relevance falls below `0.35`, the system gracefully bypasses RAG and routes to a warm fallback prompt to avoid generating hallucinated or clinical-sounding answers.
6. **LLM Generation**: Feeds the assembled prompt, sliding session history, and current context to Groq to generate a concise, warm, and structured response.

---

## 📊 Model Evaluation & Trade-offs

We evaluated multiple model architectures to optimize size, latency, accuracy, and container efficiency:

### 1. Language Detection & Emotion Classification
* **Language Detection**: We selected a **TF-IDF Vectorizer + Linear SVC** (~251MB, ~2ms latency, 98.4% F1) over XML-RoBERTa (1.1GB, ~85ms latency). It provides near-instant local classification without inflating container memory or start-up times.
* **Emotion Classification**: We chose a fine-tuned **DistilBERT** model (~268MB, ~40ms latency, 93.2% F1) over RoBERTa-large (1.4GB, ~280ms latency). This keeps CPU memory usage low enough to fit on free-tier servers while maintaining robust classification boundaries.

### 2. Retrieval-Augmented Generation (RAG)
* **Embeddings**: We selected local **`all-MiniLM-L6-v2`** (~90MB, ~5ms latency) over Cloud API embeddings (~120ms latency + billing costs). This keeps vectorization entirely local, free, and sub-10ms.
* **Vector Store**: Placed indexes in **Qdrant Cloud** to keep the local container stateless and lightweight.

### 3. Response Generation (LLM)
* **Primary LLM**: We opted for **Llama 3 70B via Groq** (~180ms latency) over local hosting (Llama 3 8B via llama.cpp takes ~4.8s on CPU) or OpenAI API (higher cost and ~350ms latency). This provides state-of-the-art responses at zero cost and ultra-low latency.

---

## 🔒 Token-Bucket Rate Limiting

To protect backend resources and Groq/Qdrant API quotas from spam or DDoS attacks, a custom **Token-Bucket Rate Limiter Middleware** is integrated.
* **Threshold**: Limits users to a maximum of 20 requests per minute.
* **Action**: Requests exceeding the limit immediately bypass pipeline execution and return an HTTP `429 Too Many Requests` error containing a user-friendly throttle message.

---

## 🐳 Containerization & Registry caching

The backend is fully containerized using a production-grade Dockerfile:
* **Multi-stage Builds**: Used to separate dependency installation from runtime execution.
* **Size Optimization**: Installed the **CPU-only PyTorch build** (`https://download.pytorch.org/whl/cpu`) instead of standard PyTorch, cutting the final Docker image size down by **2.2 GB**.
* **Layer Caching**: Structured the steps by copying `requirements.txt` and installing dependencies *before* copying application code. This ensures changes to source files do not trigger re-installation of dependencies.
* **Startup Script (`start.sh`)**: Runs the OpenTelemetry Collector in the background and launches the Uvicorn FastAPI server on the required port dynamically mapped via `${PORT:-7860}`.
* **Container Registry**: Built and pushed to **GitHub Container Registry (GHCR)** at `ghcr.io/hagar-ghazi/mental-health-mlops:latest`.

---

## 📊 System Observability (9 Metrics in Axiom)

We instrumented 9 system metrics forwarded via OpenTelemetry HTTP exporter to our OpenTelemetry Collector container, which batches and routes them directly to Axiom:

| Category | Metric Name | Type | Rationale / Reasoning |
| :--- | :--- | :--- | :--- |
| **Model / NLP** | `intent_distribution` | Counter | Tracks frequency of user intents (`greeting`, `mental_health_question`, etc.) to assess topic trends. |
| **Model / NLP** | `response_latency_ms` | Gauge | Monitors the latency of the NLP pipeline to catch response lag. |
| **Model / NLP** | `rag_retrieval_similarity_scores` | Gauge | Evaluates Qdrant similarity scores to audit database retrieval relevance. |
| **Data** | `message_length_chars` | Counter | Audits user message lengths to analyze user conversation complexity. |
| **Data** | `feedback_votes_total` | Counter | Tracks thumbs up vs. down count to audit therapist output quality. |
| **Data** | `emotion_distribution` | Counter | Profiles user emotional states to trace user base sentiment. |
| **Server** | `http_requests_total` | Counter | Records requests per route and HTTP method to monitor server load. |
| **Server** | `http_errors_total` | Counter | Monitors HTTP 4xx/5xx errors to notify of validation errors or route crashes. |
| **Server** | `active_sessions_gauge` | Gauge | Measures active memory session count to analyze container RAM consumption. |

<br/>

<div align="center">
  <img src="assets/Axiom_Dashboard.png" alt="Axiom Observability Dashboard" width="720"/>
  <p><i>Figure 2: Axiom Dashboard visualizing live metrics (Request counts, HTTP errors, response latency gauges, and intent/emotion distributions).</i></p>
</div>

### System Alerting Monitors
We also set up Axiom monitors to alert us when metrics exceed safety boundaries (e.g., elevated error rates or latency spikes):

<div align="center">
  <img src="assets/Monitors.png" alt="Axiom Alerting Monitors" width="720"/>
  <p><i>Figure 3: Axiom alerting monitors checking error rates and service latency in real-time.</i></p>
</div>

---

## 🔄 GitHub Actions CI/CD Pipeline

A continuous integration and deployment pipeline is configured in `.github/workflows/ci.yml`. On every push to the `main` branch, the workflow automates:
1. **Linting**: Verifies formatting and styling rules using **Ruff**.
2. **Testing**: Runs the **Pytest** test suite (11 unit tests covering rates, safety gates, and API endpoints).
3. **Container Build**: Compiles the Dockerfile using build caching and pushes the image to **GitHub Container Registry (GHCR)**.
4. **HF Space Deploy**: Force-pushes the code changes directly to Hugging Face Spaces, triggering a rebuild of the production space.

<div align="center">
  <img src="assets/Github_Actions.png" alt="GitHub Actions CI/CD Pipeline" width="720"/>
  <p><i>Figure 4: Automated CI/CD pipeline completing Ruff checks, Pytest runs, Docker compiles, and Hugging Face deployments.</i></p>
</div>

---

## 🔗 Project Deliverables & Links

* **Backend API Repository**: [https://github.com/Hagar-Ghazi/Mental-Health-Mlops](https://github.com/Hagar-Ghazi/Mental-Health-Mlops)
* **Deployed API Endpoint**: [https://hagarghazi-serenity-backend.hf.space](https://hagarghazi-serenity-backend.hf.space)
* **FastAPI Swagger Docs**: [https://hagarghazi-serenity-backend.hf.space/docs](https://hagarghazi-serenity-backend.hf.space/docs)
* **Frontend Client Repository**: [https://github.com/Hagar-Ghazi/Serenity-Mental-Health-Chatbot-Frontend](https://github.com/Hagar-Ghazi/Serenity-Mental-Health-Chatbot-Frontend)
* **Live Deployed Frontend (GitHub Pages)**: [https://hagar-ghazi.github.io/Serenity-Mental-Health-Chatbot-Frontend/](https://hagar-ghazi.github.io/Serenity-Mental-Health-Chatbot-Frontend/)
