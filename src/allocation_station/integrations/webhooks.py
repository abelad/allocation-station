"""Webhook Notifications Module."""

import requests
from typing import Dict, List
from datetime import datetime


class WebhookManager:
    """Manage webhook notifications."""

    def __init__(self):
        """Initialize webhook manager."""
        self.endpoints = []

    def add_endpoint(self, url: str):
        """Add webhook endpoint."""
        self.endpoints.append(url)

    def send_notification(self, event_type: str, data: Dict) -> bool:
        """Send webhook notification."""
        payload = {
            "event_type": event_type,
            "timestamp": datetime.now().isoformat(),
            "data": data
        }

        success = True
        for endpoint in self.endpoints:
            try:
                response = requests.post(endpoint, json=payload, timeout=5)
                if response.status_code != 200:
                    success = False
            except Exception as e:
                print(f"Webhook failed for {endpoint}: {e}")
                success = False

        return success
