"""Send a small trace to the local Jaeger and read it back.

    docker compose -f examples/otel_jaeger/docker-compose.yml up -d
    pip install "llm-logs[otel]"
    python examples/otel_jaeger/send_traces.py
    # then open http://localhost:16686 and pick the service "llm-logs-demo"

No API key and no network beyond localhost: the "model" is a stand-in. To use
another backend, leave this script alone and set the standard variables, e.g.

    export OTEL_EXPORTER_OTLP_ENDPOINT=https://your-backend.example/api/public/otel
    export OTEL_EXPORTER_OTLP_HEADERS="Authorization=Basic <credentials>"
"""

from __future__ import annotations

import json
import os
import time
import urllib.request
from typing import Any

import llm_logs as ll

SERVICE = "llm-logs-demo"
os.environ.setdefault("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318")
JAEGER_API = os.environ.get("JAEGER_QUERY_URL", "http://localhost:16686")


@ll.trace(provider="demo", operation="chat")
def ask(messages: list[dict[str, str]], *, model: str, temperature: float = 0.0) -> dict[str, str]:
    time.sleep(0.05)
    return {"role": "assistant", "content": "Refunds are issued within 14 days."}


def main() -> None:
    # capture_content=True here because everything stays on this machine.
    ll.configure(sinks=[ll.OtelSink(service_name=SERVICE, capture_content=True)])
    with ll.span("rag_pipeline", session_id="demo-session", metadata={"route": "/chat"}) as root:
        with ll.span("retrieve", operation="retrieval"):
            time.sleep(0.01)
        ask([{"role": "user", "content": "How long do refunds take?"}], model="demo-model")
    trace_id = root.trace_id
    ll.shutdown(15)
    stats = ll.stats()
    print("sent:", stats)
    if stats.failed or stats.written != 3:
        raise SystemExit(
            "The spans could not be exported. Is the backend running?\n"
            "  docker compose -f examples/otel_jaeger/docker-compose.yml up -d"
        )

    traces: list[dict[str, Any]] = []
    for _ in range(20):  # Jaeger indexes asynchronously
        try:
            with urllib.request.urlopen(f"{JAEGER_API}/api/traces/{trace_id}", timeout=5) as reply:
                traces = json.load(reply).get("data") or []
        except OSError:
            traces = []
        if traces and len(traces[0]["spans"]) == 3:
            break
        time.sleep(0.5)
    else:
        raise SystemExit("The spans were exported, but the trace did not show up in Jaeger.")

    by_id = {span["spanID"]: span for span in traces[0]["spans"]}
    for span in traces[0]["spans"]:
        parent = next((r["spanID"] for r in span["references"] if r["refType"] == "CHILD_OF"), None)
        tags = {t["key"]: t["value"] for t in span["tags"] if t["key"].startswith("gen_ai.")}
        under = f"  (child of {by_id[parent]['operationName']})" if parent else ""
        print(f"- {span['operationName']}{under}")
        for key in sorted(tags):
            print(f"    {key} = {str(tags[key])[:90]}")
    print(f"\nopen {JAEGER_API}/trace/{trace_id}")


if __name__ == "__main__":
    main()
