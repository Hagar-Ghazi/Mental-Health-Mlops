import logging
from opentelemetry import metrics
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.sdk.resources import Resource
from app.config import OTEL_EXPORTER_OTLP_ENDPOINT, OTEL_SERVICE_NAME, ENABLE_TELEMETRY

logger = logging.getLogger("app_logger")


# Safe No-Op Fallbacks for Metrics if initialization fails
class DummyCounter:
    def add(self, value, attributes=None):
        pass


class DummyHistogram:
    def record(self, value, attributes=None):
        pass


# Global Metric Instances (Pre-initialized with Dummy fallbacks)
intent_counter = DummyCounter()
latency_histogram = DummyHistogram()
rag_scores_histogram = DummyHistogram()
msg_length_histogram = DummyHistogram()
feedback_counter = DummyCounter()
emotion_counter = DummyCounter()
http_requests_counter = DummyCounter()
http_errors_counter = DummyCounter()


def init_metrics():
    """Initializes the OpenTelemetry meter provider, registers exporter and metrics."""
    global intent_counter, latency_histogram, rag_scores_histogram
    global msg_length_histogram, feedback_counter, emotion_counter
    global http_requests_counter, http_errors_counter

    if not ENABLE_TELEMETRY:
        logger.info(
            "OpenTelemetry is disabled (ENABLE_TELEMETRY=False). Using dummy metrics."
        )
        return

    try:
        resource = Resource(attributes={"service.name": OTEL_SERVICE_NAME})

        # Build explicit HTTP endpoint for OTLP metric receiver
        # e.g., http://localhost:4318 -> http://localhost:4318/v1/metrics
        metrics_endpoint = f"{OTEL_EXPORTER_OTLP_ENDPOINT.rstrip('/')}/v1/metrics"

        exporter = OTLPMetricExporter(endpoint=metrics_endpoint)
        reader = PeriodicExportingMetricReader(exporter, export_interval_millis=5000)
        provider = MeterProvider(resource=resource, metric_readers=[reader])

        metrics.set_meter_provider(provider)
        meter = metrics.get_meter("serenity_meter")

        # 1. Model / NLP Metrics
        intent_counter = meter.create_counter(
            name="intent_distribution",
            description="Counts detected user intents (greeting, goodbye, mental_health_question, out_of_scope)",
            unit="1",
        )
        latency_histogram = meter.create_histogram(
            name="response_latency_ms",
            description="Response latency of the chatbot NLP pipeline in milliseconds",
            unit="ms",
        )
        rag_scores_histogram = meter.create_histogram(
            name="rag_retrieval_similarity_scores",
            description="Similarity scores of context retrieved from Qdrant vector database",
            unit="score",
        )

        # 2. Data Metrics
        msg_length_histogram = meter.create_histogram(
            name="message_length_chars",
            description="Distribution of user input query character counts",
            unit="char",
        )
        feedback_counter = meter.create_counter(
            name="feedback_votes_total",
            description="Total count of helpful (up) or unhelpful (down) user feedback",
            unit="1",
        )
        emotion_counter = meter.create_counter(
            name="emotion_distribution",
            description="Counts detected user emotions (sadness, joy, fear, etc.)",
            unit="1",
        )

        # 3. Server Metrics
        http_requests_counter = meter.create_counter(
            name="http_requests_total",
            description="Total count of incoming HTTP requests to server routes",
            unit="1",
        )
        http_errors_counter = meter.create_counter(
            name="http_errors_total",
            description="Total count of server HTTP requests resulting in 4xx or 5xx status codes",
            unit="1",
        )

        # Observable Gauge for Active Session Count
        from app.services.session import session_store

        def observe_active_sessions(options):
            return [metrics.Observation(session_store.active_count())]

        meter.create_observable_gauge(
            name="active_sessions_gauge",
            callbacks=[observe_active_sessions],
            description="Number of concurrent active chat sessions in memory",
        )

        logger.info("OpenTelemetry metrics successfully initialized.")

    except Exception as e:
        logger.error(
            f"OpenTelemetry metrics failed to initialize, running with dummy metrics: {e}",
            exc_info=True,
        )
