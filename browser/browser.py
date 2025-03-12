from PyQt6.QtCore import QUrl
from PyQt6.QtWebEngineCore import QWebEngineProfile, QWebEngineSettings
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLineEdit

from browser.basic_sec_interceptor import SecurityInterceptor


class BrowserApp(QWidget):
    """Eenvoudige webbrowser met PyQtWebEngine"""
    def __init__(self):
        super().__init__()
        # Main layout
        self.layout = QVBoxLayout(self)

        # Create navigation bar layout
        nav_bar = QHBoxLayout()

        # Create buttons
        self.btn_back = QPushButton("◀")
        self.btn_back.clicked.connect(self.go_back)

        self.btn_forward = QPushButton("▶")
        self.btn_forward.clicked.connect(self.go_forward)

        self.btn_reload = QPushButton("⟳")
        self.btn_reload.clicked.connect(self.reload_page)

        self.btn_home = QPushButton("🏠")
        self.btn_home.clicked.connect(self.go_home)

        # URL Bar
        self.url_bar = QLineEdit()
        self.url_bar.setPlaceholderText("Enter URL and press Enter")
        self.url_bar.returnPressed.connect(self.load_url)

        # Add widgets to navigation bar
        nav_bar.addWidget(self.btn_back)
        nav_bar.addWidget(self.btn_forward)
        nav_bar.addWidget(self.btn_reload)
        nav_bar.addWidget(self.btn_home)
        nav_bar.addWidget(self.url_bar)

        # Create browser widget
        self.browser = QWebEngineView()

        # Secure browser profile
        # profile = QWebEngineProfile.defaultProfile()
        # profile.setPersistentCookiesPolicy(
        #     QWebEngineProfile.PersistentCookiesPolicy.NoPersistentCookies)  # No cookie storage
        # profile.setHttpCacheType(QWebEngineProfile.HttpCacheType.NoCache)  # No caching
        # profile.setPersistentStoragePath("")  # Disable storage
        # profile.setSpellCheckEnabled(False)  # Disable spell check tracking
        # profile.setUrlRequestInterceptor(SecurityInterceptor())  # Block tracking requests
        #
        # # Web settings for privacy
        # settings = self.browser.settings()
        # settings.setAttribute(QWebEngineSettings.WebAttribute.JavascriptEnabled, False)  # Disable JavaScript
        # settings.setAttribute(QWebEngineSettings.WebAttribute.LocalStorageEnabled, False)  # No local storage
        # settings.setAttribute(QWebEngineSettings.WebAttribute.PluginsEnabled, False)  # Disable plugins
        # settings.setAttribute(QWebEngineSettings.WebAttribute.AutoLoadImages,
        #                       True)  # Load images but block other trackers

        self.browser.setUrl(QUrl("https://www.duckduckgo.com"))  # Default homepage
        self.browser.urlChanged.connect(self.update_url_bar)

        # Add navigation bar and browser to main layout
        self.layout.addLayout(nav_bar)
        self.layout.addWidget(self.browser)

    def load_url(self):
        url_text = self.url_bar.text()
        if not url_text.startswith(("http://", "https://")):
            url_text = "https://" + url_text  # Auto-add https:// if missing
        self.browser.setUrl(QUrl(url_text))

    def go_home(self):
        self.browser.setUrl(QUrl("https://www.duckduckgo.com"))  # Default home page

    def go_back(self):
        self.browser.back()

    def go_forward(self):
        self.browser.forward()

    def reload_page(self):
        self.browser.reload()
    def update_url_bar(self, qurl):
        self.url_bar.setText(qurl.toString())