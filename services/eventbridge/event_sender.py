"""XProtect AnalyticsEvent sender scaffold."""
from __future__ import annotations

import os
from typing import Dict


def build_analytics_event_xml(
    source_id: str,
    event_type: str,
    properties: Dict[str, str],
) -> str:
    props_xml = "".join(
        f"<Property><Key>{k}</Key><Value>{v}</Value></Property>" for k, v in properties.items()
    )
    return (
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>"
        "<AnalyticsEvent>"
        f"<SourceID>{source_id}</SourceID>"
        f"<EventType>{event_type}</EventType>"
        f"<Properties>{props_xml}</Properties>"
        "</AnalyticsEvent>"
    )


def send_event(xml_payload: str) -> None:
    try:
        import requests  # type: ignore
    except Exception as exc:
        raise RuntimeError("requests is required to send events.") from exc

    endpoint = os.environ.get("EVENT_SERVER_URL", "")
    if not endpoint:
        raise RuntimeError("Set EVENT_SERVER_URL to XProtect Event Server endpoint.")

    resp = requests.post(endpoint, data=xml_payload, headers={"Content-Type": "text/xml"}, timeout=10)
    resp.raise_for_status()


def main() -> int:
    xml_payload = build_analytics_event_xml(
        source_id="CAMERA_1",
        event_type="AnimalDetected",
        properties={"species": "unknown", "confidence": "0.5"},
    )
    send_event(xml_payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
