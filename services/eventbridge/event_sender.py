"""XProtect Analytics Event sender.

Builds the XML that the XProtect Event Server accepts on its Analytics Events
port (9090 by default) and posts it over HTTP.

Setup required on the XProtect side (Management Client):
1. Tools > Options > Analytics Events: enable, and allow this machine's address.
2. Rules and Events > Analytics Events: create an event whose *name* equals the
   ``event_type`` used here (for example ``AnimalBehavior.Pacing``). The Event
   Server drops events whose ``Message`` does not match a defined event.
3. ``source_id`` must be the camera's name or address as XProtect knows it.

NOTE: the element layout follows Milestone's published AnalyticsEvent schema
(``urn:milestone-systems``) but has not been exercised against a live Event
Server from this repository. Verify with one test event before relying on it.
"""
from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone
from typing import Dict, Optional
from xml.sax.saxutils import escape

MILESTONE_NS = "urn:milestone-systems"


def build_analytics_event_xml(
    source_id: str,
    event_type: str,
    properties: Optional[Dict[str, str]] = None,
    *,
    timestamp: Optional[datetime] = None,
    event_id: Optional[str] = None,
    description: Optional[str] = None,
) -> str:
    """Build an AnalyticsEvent document.

    ``properties`` are flattened into the ``CustomTag`` header field as
    ``key=value;key=value`` (searchable in the Smart Client alarm list) and into
    ``Description`` unless an explicit description is given. All values are
    XML-escaped.
    """
    properties = properties or {}
    when = timestamp or datetime.now(tz=timezone.utc)
    if when.tzinfo is None:
        raise ValueError("timestamp must be timezone-aware")
    tag = ";".join(f"{k}={v}" for k, v in properties.items())
    desc = description if description is not None else tag

    return (
        '<?xml version="1.0" encoding="utf-8"?>'
        '<AnalyticsEvent xmlns:i="http://www.w3.org/2001/XMLSchema-instance" '
        f'xmlns="{MILESTONE_NS}">'
        "<EventHeader>"
        f"<ID>{escape(event_id or str(uuid.uuid4()))}</ID>"
        f"<Timestamp>{escape(when.isoformat())}</Timestamp>"
        f"<Type>{escape(str(event_type))}</Type>"
        f"<Message>{escape(str(event_type))}</Message>"
        f"<CustomTag>{escape(tag)}</CustomTag>"
        f"<Source><Name>{escape(str(source_id))}</Name></Source>"
        "</EventHeader>"
        f"<Description>{escape(desc)}</Description>"
        "<Vendor><Name>AnimalUniversity</Name></Vendor>"
        "</AnalyticsEvent>"
    )


def send_event(xml_payload: str, endpoint: Optional[str] = None, timeout: float = 10.0) -> None:
    try:
        import requests  # type: ignore
    except Exception as exc:
        raise RuntimeError("requests is required to send events.") from exc

    endpoint = endpoint or os.environ.get("EVENT_SERVER_URL", "")
    if not endpoint:
        raise RuntimeError(
            "Set EVENT_SERVER_URL to the XProtect Event Server analytics endpoint, "
            "for example http://<event-server>:9090/"
        )

    resp = requests.post(
        endpoint,
        data=xml_payload.encode("utf-8"),
        headers={"Content-Type": "text/xml; charset=utf-8"},
        timeout=timeout,
    )
    resp.raise_for_status()


def main() -> int:
    xml_payload = build_analytics_event_xml(
        source_id=os.environ.get("XPROTECT_SOURCE", "CAMERA_1"),
        event_type="AnimalBehavior.Pacing",
        properties={"species": "unknown", "score": "0.5", "test": "true"},
    )
    send_event(xml_payload)
    print("Test event sent.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
