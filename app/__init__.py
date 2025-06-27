from flask import Flask
from .object_detection import object_detection
from .obstacle_warning import obstacle_warning
from .scan_text import scan_text
from .localize import localize

def create_app():
    app = Flask(__name__)
    app.register_blueprint(object_detection)
    app.register_blueprint(obstacle_warning)
    app.register_blueprint(scan_text)
    app.register_blueprint(localize)
    
    return app