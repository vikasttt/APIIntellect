"""Logging and telemetry configuration using Azure Monitor + structlog."""
from __future__ import annotations
import logging
import sys
import structlog
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
def configure_logging(debug: bool = False) -> None:
    """Configure structlog with JSON output."""
    log_level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=log_level,
    )
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
def configure_telemetry(
    connection_string: str | None,
    service_name: str,
    service_version: str,
) -> None:
    """Configure Azure Monitor OpenTelemetry if connection string is set."""
    if not connection_string:
        return
    try:
        from azure.monitor.opentelemetry import configure_azure_monitor
        configure_azure_monitor(
            connection_string=connection_string,
        )
        resource = Resource.create(
            {
                "service.name": service_name,
                "service.version": service_version,
            }
        )
        provider = TracerProvider(resource=resource)
        trace.set_tracer_provider(provider)
        structlog.get_logger(__name__).info(
            "Azure Monitor telemetry configured",
            service_name=service_name,
            service_version=service_version,
        )
    except ImportError:
        structlog.get_logger(__name__).warning(
            "azure-monitor-opentelemetry not installed; skipping telemetry"
        )
def get_logger(name: str) -> structlog.BoundLogger:
    """Return a bound structlog logger."""
    return structlog.get_logger(name)