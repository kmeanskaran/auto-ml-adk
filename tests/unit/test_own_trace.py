"""A background job started inside an unsampled request still gets traced.

Deployed, Google's front door samples few requests; a pipeline run started inside an
unsampled one used to inherit that and send no spans to Cloud Trace.
"""

from __future__ import annotations

import asyncio

from opentelemetry import trace as otel
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.sampling import ALWAYS_ON, ParentBased
from opentelemetry.trace import NonRecordingSpan, SpanContext, TraceFlags

from app.harness.trace import own_trace

tracer = TracerProvider(sampler=ParentBased(ALWAYS_ON)).get_tracer("test")


async def job() -> bool:
    with tracer.start_as_current_span("invoke_workflow") as span:
        return span.is_recording()


async def inside_unsampled_request(work):
    request = SpanContext(
        trace_id=0x1, span_id=0x2, is_remote=True, trace_flags=TraceFlags(0)
    )
    with otel.use_span(NonRecordingSpan(request)):
        return await asyncio.create_task(work())


def test_a_job_inherits_the_requests_decision_without_own_trace():
    assert asyncio.run(inside_unsampled_request(job)) is False


def test_own_trace_records_the_job_anyway():
    assert asyncio.run(inside_unsampled_request(lambda: own_trace(job()))) is True
