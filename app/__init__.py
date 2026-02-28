from flask import Flask
from .object_detection import object_detection
from .midas_with_object_detect import obstacle_detection
from .ar_markers_navigation import ar_markers_navigation
from .ar_markers_navigation_test import ar_markers_navigation_test
from .scan_text import scan_text

def create_app():
    app = Flask(__name__)
    app.register_blueprint(object_detection)
    app.register_blueprint(obstacle_detection)
    app.register_blueprint(ar_markers_navigation)
    app.register_blueprint(ar_markers_navigation_test)
    app.register_blueprint(scan_text)
    
    return app
