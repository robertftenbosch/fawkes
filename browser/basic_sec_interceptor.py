from PyQt6.QtCore import QUrl
from PyQt6.QtWebEngineCore import QWebEngineProfile, QWebEngineUrlRequestInterceptor


class SecurityInterceptor(QWebEngineUrlRequestInterceptor):
    """Blocks tracking scripts and enforces HTTPS"""

    def interceptRequest(self, info):
        url = info.requestUrl().toString()

        # Block known tracking domains
        tracking_domains = ["google-analytics.com", "doubleclick.net", "adservice.google.com"]
        if any(tracker in url for tracker in tracking_domains):
            info.block(True)

        # Enforce HTTPS
        if url.startswith("http://"):
            secure_url = url.replace("http://", "https://", 1)
            info.redirect(QUrl(secure_url))