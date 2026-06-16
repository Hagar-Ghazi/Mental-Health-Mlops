---
title: Serenity Backend
emoji: 🪐
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# 🪐 Serenity Chatbot: High-Performance Empathetic Mental Health AI Support Platform

Serenity is an empathetic mental health support chatbot system designed to deliver immediate, clinically guided and safe conversational assistance to users. This repository houses the production-ready **Asynchronous FastAPI Backend API** containerized with Docker monitored via OpenTelemetry and Axiom and deployed automatically through a GitHub Actions CI/CD pipeline.

---
## 🎨 User Interface & Frontend Design

The Serenity client is a modern, responsive web interface built to foster a calming and premium user experience.

<div align="center">
  <img src="assets/Serenity_Frontend.png" alt="Serenity Frontend Chat Interface" width="720"/>
  <p><i>Figure 1: Premium frontend interface demonstrating responsive message bubbles, typing indicators and user feedback buttons.</i></p>
</div>

---

## 📋 Project Planning & API Contract Analysis

At the project inception we conducted a thorough planning phase to map requirements and integrate backend and frontend systems seamlessly.



### 1. Frontend Code Review
We analyzed the client application (composed of `index.html`, `style.css`, and `app.js`) to establish the interface logic:

- The frontend exposes a sliding conversation pane displaying incoming and outgoing messages.
- Message rendering relies on structured JSON fields (`response`, `intent`, `emotion`, `crisis_flag`).
- Thumbs up/down icons map to an interactive feedback loop, pushing rating events back to the backend.



### 2. API Contract Specification
Based on the client's execution flow we defined a strict, low-latency API contract:

| Endpoint | HTTP Method | Request Payload | Response Payload | Description |
| :--- | :--- | :--- | :--- | :--- |
| `/chat` | **POST** | `{"message": "string"}` | `{"response": "string", "answer": "string", "session_id": "string", "emotion": "string", "emotion_conf": float, "language": "string", "intent": "string", "crisis_flag": boolean}` | Orchestrates the RAG, emotion, and LLM pipelines to generate a response. |
| `/feedback` | **POST** | `{"vote": "string", "user_message": "string", "bot_response": "string"}` | `{"status": "success", "message": "Feedback saved successfully"}` | Registers thumbs-up/down ratings and logs them to SQLite. |
| `/health` | **GET** | *None* | `{"status": "ok", "active_sessions": int}` | Reports application health and tracks concurrent memory sessions. |
| `/health/timings` | **GET** | *None* | `{"last_runs": list}` | Debug route displaying timing breakdowns of recent pipeline runs. |



- **CORS Configuration**:
  The backend implements standard `CORSMiddleware` configured to allow all origins (`*`) during student development ensuring cross-origin requests from the GitHub Pages frontend origin bypass browser security blocks successfully.

---



## 🧠 NLP Pipeline & Decision Logic Architecture

Serenity's NLP orchestration engine routes queries through a multi-stage, safety-first workflow designed to prevent hallucinations, identify crises and deliver hyper-personalized support.

```
                   [ User Message ]
                           │
                           ▼
             ┌───────────────────────────┐
             │ 1. Language Lock Detection│  <-- Local TF-IDF + Linear SVC Classifier
             └─────────────┬─────────────┘
                           ▼
             ┌───────────────────────────┐
             │ 2. Emotion Classification │  <-- Local PyTorch DistilBERT Classifier
             └─────────────┬─────────────┘
                           ▼
             ┌───────────────────────────┐
             │ 3. Safety Check / Intent  │  <-- Hardcoded Triggers + Groq LLM Classifier
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
             │ 4. Prompt Engineering     │  <-- Combines Context, History and Emotion
             └─────────────┬─────────────┘
                           ▼
             ┌───────────────────────────┐
             │ 5. Response Generation    │  <-- Groq Llama 3 70B LLM Completion
             └───────────────────────────┘
```


### 1. Language Lock
Evaluates whether the incoming query is in English or Arabic to enforce language consistency. If Arabic is detected Uvicorn locks the output language to warm Egyptian Arabic (`عامية مصرية بسيطة`) to provide a localized and comfortable conversational experience.


### 2. Emotion Profiling
Queries run locally through a fine-tuned DistilBERT sequence classifier to evaluate emotional state (`sadness`, `fear`, `anger`, `joy`, `love`, `surprise`).



### 3. Crisis & Safety First-Line Defense
The system implements a critical, multi-tiered safety checking system:
- **Hardcoded Triggers**: Scans the input text for crisis keywords (e.g., suicide, self-harm, ending my life).
- **LLM Safety Intent**: Groq Llama 3 parses the message to catch complex safety breaches.
- **Emotion Risk**: Flags messages exhibiting extreme despair or high-confidence distress signals.
- **Crisis Routing**: If flagged the pipeline immediately bypasses RAG and general conversational routing generates an empathetic crisis intervention response and injects emergency hotline numbers dynamically tailored to the user's location based on their IP address (e.g., Egypt hotlines for Egyptian IPs, US/International hotlines for others).


### 4. Intent Classification
Using few-shot instructions Groq Llama 3 categorizes queries into **greeting, goodbye, mental_health_question, or out_of_scope**.

- **Out-of-Scope Handling**:
  Queries categorized as out-of-scope (e.g., asking for python scripts or recipe instructions) immediately bypass downstream RAG retrieval and LLM steps. They return a polite boundary-setting response protecting resources and preventing API usage charges.


### 5. Adaptive Context Retrieval (RAG)
For valid mental health questions the user query is encoded locally via **sentence-transformers/all-MiniLM-L6-v2** and searched in Qdrant Cloud.

An **Adaptive Top-K** mechanism adjusts retrieval density (e.g., top-3 for short inputs, top-7 for detailed entries) to balance search coverage and context window size.



### 6. Emotion-Based Context Reranking
Retrieved chunks are reranked using a custom heuristic: similarity scores are boosted if a chunk's topic matches the user's detected emotional state (e.g. boosting depression topics for sad users) or if the chunk exhibits high empathy tags.


### 7. Intelligence Heuristic / Decision Gate
The pipeline reviews the top search scores. If the best similarity score falls below a threshold of 0.35 the RAG context is bypassed entirely. The system routes to a warm fallback prompt to prevent generating hallucinated or clinical-sounding answers.


### 8. Context-Aware LLM Generation
The final response is generated using Groq Llama 3 70B combining the compiled therapist system prompt, retrieved clinical context, sliding session history (up to 12 turns to prevent memory overflow) and the user query.

---


## 📊 Model Evaluation & Engineering Trade-offs

To optimize size, inference latency, accuracy and server memory consumption, we evaluated multiple model architectures:

### Components Comparison Table

| Pipeline Stage | Evaluated Candidate | Size on Disk | Latency | Accuracy / F1 | Hosting Cost | GIL / Threading Impact | Decision & Rationale |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Language Detector** | XML-RoBERTa (Cloud/API) | ~1.1 GB | ~85 ms | 99.1% F1 | Medium (API fees) | None (Network) | ❌ Bypassed due to container inflation and cost. |
| **Language Detector** | **TF-IDF + Linear SVC** | **~251 MB** | **~2 ms** | **98.4% F1** | **$0.00 (Local)** | **Low (Offloaded)** | **✅ Selected**: Instant local classification with zero CPU overhead. |
| **Emotion Classifier** | RoBERTa-large (Fine-tuned) | ~1.4 GB | ~280 ms | 94.8% F1 | High | High (Blocks GIL) | ❌ Bypassed due to high latency and heavy RAM footprint. |
| **Emotion Classifier** | **DistilBERT (Fine-tuned)** | **~268 MB** | **~40 ms** | **93.2% F1** | **$0.00 (Local)** | **Medium (Offloaded)** | **✅ Selected**: Excellent accuracy-to-size ratio; runs efficiently on free-tier CPUs. |
| **Vector Embeddings** | OpenAI text-embedding-ada-002 | *None* | ~120 ms | N/A | Variable (Token bill) | None (Network) | ❌ Bypassed due to network overhead and token billing. |
| **Vector Embeddings** | **all-MiniLM-L6-v2 (Local)** | **~90 MB** | **~5 ms** | **N/A** | **$0.00 (Local)** | **Low (Offloaded)** | **✅ Selected**: Fast, local vectorization with no external API dependency. |
| **Response Generation** | Llama 3 8B (Local CPU) | ~4.8 GB | ~4.8 s | Moderate | $0.00 | High (Locks Event Loop) | ❌ Bypassed due to unacceptable latency and memory requirements. |
| **Response Generation** | **Llama 3 70B (via Groq)** | ***Stateless*** | **~180 ms** | **State-of-the-Art** | **$0.00 (Free Tier)** | **None (Network Async)**| **✅ Selected**: Exceptional conversational quality and sub-200ms latency. |

---

## High-Performance Asynchronous Architecture

Initially the backend codebase ran a synchronous thread-blocking pipeline. In that configuration, network API calls (Qdrant vector search, Groq completions) and CPU-heavy classifier inferences ran sequentially locking Uvicorn's execution thread and resulting in high latencies (30–60+ seconds per message).

To resolve this bottleneck we refactored the entire stack to be **fully asynchronous (Async/Await)**:

1. **Async Network Clients**:
   Refactored the database and LLM pipelines to use **AsyncQdrantClient and **AsyncGroq** ensuring that external network wait times do not lock the server threads.

2. **Concurrent CPU Tasks**:
   Leveraged **fastapi.concurrency.run_in_threadpool** wrapped inside asyncio.gather to execute the local scikit-learn language detector and fine-tuned PyTorch emotion classifier concurrently in non-blocking worker threads.

3. **Async Endpoints**:
   Promoted **/chat** and **/feedback** API routes to async def and wrapped SQLite database transactions inside asynchronous worker thread pools.

**Result**:
 Average message processing latency dropped from 50 seconds to **1.65 seconds **achieving a **30x performance acceleration** under live conditions.

---

## Security & Resource Management: Rate Limiting

To protect backend resources, database capacity and downstream API quotas (Groq & Qdrant Cloud) we built a custom **Sliding-Window Token-Bucket Rate Limiter Middleware**:

- **Threshold**:
  Restricts each unique IP address to a maximum of **20 requests per minute**.

- **Action**:
  Requests exceeding the limit bypass pipeline execution entirely return an HTTP **429 Too Many Requests** status and deliver a user-friendly message: **"You're sending messages too quickly. Please wait a moment and try again."**

- **Exemptions**:
  Health checking endpoints (/health) bypass the rate limiter to maintain uptime reporting services.

---

## Logging Infrastructure

We set up a comprehensive logging system using Python's native logging module:

- **Console Logging**:
  Configured with standardized formatting and distinct severity levels (INFO, WARNING, ERROR) to trace server health

- **Offline Analytics logging**:
  Every chat query, pipeline output, classification confidence score, retrieval score and execution timeline is captured in a structured JSON Lines file (**logs/pipeline_conversations.jsonl**) this enables offline audit logs and telemetry parsing without introducing database latency.

---

## Unit Testing Suite & Code Quality Control

### 1. Automated Tests
We wrote an extensive unit testing suite under **tests/** utilizing Pytest to validate key application flows:

- **Happy Paths**:
  Confirms successful interactions for greetings, mental health questions, feedback submissions, and health checks.

- **Edge Cases**:
  Validates pipeline robustness against empty queries and out-of-scope inputs.

- **Safety Overrides**:
  Ensures crisis keywords trigger immediate safety protocols and hotline injections.

- **Rate Limiting**:
  Verifies that sending more than 20 queries per minute triggers HTTP 429 errors.

We mocked Qdrant and Groq API clients to keep tests fast and independent of network availability.

```bash
# Command to execute tests
python -m pytest -v
```

*All 11 unit tests pass successfully in less than a second:*

```
tests\test_chat.py ....                                                  [ 36%]
tests\test_feedback.py ..                                                [ 54%]
tests\test_health.py ..                                                  [ 72%]
tests\test_pipeline.py ...                                               [100%]
======================== 11 passed, 1 warning in 0.61s ========================
```


### 2. Pre-commit Hooks & Linting
A `.pre-commit-config.yaml` configuration is configured with **Ruff** to enforce code quality before commits:

- **Linter**:
  `ruff` automatically analyzes files for unused imports and syntax errors.

- **Formatter**:
  `ruff-format` ensures code formatting complies with PEP 8 standards.

---

## 🐳 Containerization & Caching Strategy

The backend is fully containerized using a production-grade Dockerfile:
- **Multi-Stage Build**:
  Separates compile-time dependencies from the runtime image to reduce size.

- **Dependency Caching**: 
  Dependencies are installed *before* copying the application source code. This ensures changes to python files do not invalidate the cached dependency layers, accelerating build times.

- **Size Optimization**:
  Installs the CPU-only PyTorch build (https://download.pytorch.org/whl/cpu) instead of standard PyTorch cutting the final Docker image size down by **2.2 GB**.

- **Service Orchestration**:
  An entrypoint script (start.sh) launches the OpenTelemetry Collector in the background and starts the Uvicorn FastAPI server on the dynamically mapped Hugging Face port (${PORT:-7860}).


### Docker Layer Caching Verification
Docker build logs verify that layer caching is functioning correctly. Build stages utilize cached layers (indicated by `CACHED` tags in CI execution logs) reducing pipeline run times from several minutes to under 30 seconds.

<div align="center">
  <img src="assets/Serenity_backend.png" alt="Docker Build Cache Hits" width="720"/>
  <p><i>Figure 2: GitHub Actions Docker compile stage confirming cached layers are utilized for faster builds.</i></p>
</div>

---

## System Observability (9 Metrics in Axiom)

We instrumented 9 system metrics forwarded via OpenTelemetry HTTP exporter to our OpenTelemetry Collector container which batches and routes them directly to Axiom:

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
  <p><i>Figure 3: Axiom Dashboard visualizing live metrics (Request counts, HTTP errors, response latency gauges, and intent/emotion distributions).</i></p>
</div>


### System Alerting Monitors
Axiom monitors are configured to alert the development team when metrics exceed safety thresholds such as elevated error rates or high latency.

<div align="center">
  <img src="assets/Monitors.png" alt="Axiom Alerting Monitors" width="720"/>
  <p><i>Figure 4: Axiom alerting monitors checking error rates and service latency in real-time.</i></p>
</div>

---


## GitHub Actions CI/CD Pipeline

A continuous integration and deployment pipeline is configured in `.github/workflows/ci.yml`. On every push to the `main` branch the workflow automates:

1. **Linting**:
   Verifies formatting and styling rules using **Ruff**.

2. **Testing**:
   Runs the **Pytest** test suite (11 unit tests covering rates, safety gates, and API endpoints).

3. **Container Build**:
   Compiles the Dockerfile using build caching and pushes the image to **GitHub Container Registry (GHCR)**.

4. **HF Space Deploy**:
   Force-pushes the code changes directly to Hugging Face Spaces, triggering a rebuild of the production space.

<div align="center">
  <img src="assets/Github_Actions.png" alt="GitHub Actions CI/CD Pipeline" width="720"/>
  <p><i>Figure 5: Automated CI/CD pipeline completing Ruff checks, Pytest runs, Docker compiles and Hugging Face deployments.</i></p>
</div>

---

## 🔗 Project Deliverables & Links

- **Backend API Repository**: [https://github.com/Hagar-Ghazi/Mental-Health-Mlops](https://github.com/Hagar-Ghazi/Mental-Health-Mlops)
- **Deployed API Endpoint**: [https://hagarghazi-serenity-backend.hf.space](https://hagarghazi-serenity-backend.hf.space)
- **FastAPI Swagger Docs**: [https://hagarghazi-serenity-backend.hf.space/docs](https://hagarghazi-serenity-backend.hf.space/docs)
- **Frontend Client Repository**: [https://github.com/Hagar-Ghazi/Serenity-Mental-Health-Chatbot-Frontend](https://github.com/Hagar-Ghazi/Serenity-Mental-Health-Chatbot-Frontend)
- **Live Deployed Frontend (GitHub Pages)**: [https://hagar-ghazi.github.io/Serenity-Mental-Health-Chatbot-Frontend/](https://hagar-ghazi.github.io/Serenity-Mental-Health-Chatbot-Frontend/)
