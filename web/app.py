"""
Flask web server with Server-Sent Events (SSE) for real-time aggregated count updates.
Provides REST endpoints for system health, aggregated stats, and per-camera stats.
Multi-camera dashboard support via vanilla JS with dynamic camera cards.
"""

import logging
import json
import time
from functools import wraps
from typing import Iterator
from flask import Flask, render_template, Response, jsonify

import config

logger = logging.getLogger(__name__)

app = Flask(__name__, template_folder='templates')
app.config['JSON_SORT_KEYS'] = False

camera_manager = None


def set_camera_manager(manager) -> None:
    """
    Inject the CameraManager instance into the Flask app.
    Called from main.py after manager is created and started.

    Args:
        manager: CameraManager instance
    """
    global camera_manager
    camera_manager = manager
    logger.info("CameraManager instance set in Flask app")


@app.route('/')
def index():
    """Serve the main dashboard HTML."""
    return render_template('index.html')


@app.route('/stream')
def stream():
    """
    Server-Sent Events endpoint for real-time count updates.
    Streams aggregated counts from all cameras.
    Format: {total_in, total_out, current_inside, camera_count, timestamp}

    Yields:
        SSE formatted data string with JSON payload
    """
    if camera_manager is None:
        logger.error("CameraManager instance not initialized")
        return Response("error: camera_manager not initialized", status=500, mimetype='text/plain')

    def generate_events() -> Iterator[str]:
        """Generate SSE events with aggregated count updates."""
        try:
            logger.info("SSE client connected")
            last_stats = None

            while True:
                try:
                    current_stats = camera_manager.get_aggregated_stats()

                    if current_stats != last_stats:
                        data = {
                            'total_in': current_stats['total_in'],
                            'total_out': current_stats['total_out'],
                            'current_inside': current_stats['current_inside'],
                            'camera_count': current_stats['camera_count'],
                            'connected_count': current_stats['connected_count'],
                            'timestamp': time.time(),
                        }

                        yield f"data: {json.dumps(data)}\n\n"
                        last_stats = current_stats

                    time.sleep(config.SSE_UPDATE_INTERVAL)

                except Exception as e:
                    logger.error("Error in SSE stream: %s", str(e))
                    yield f"data: {json.dumps({'error': str(e)})}\n\n"
                    time.sleep(1)

        except GeneratorExit:
            logger.info("SSE client disconnected")
        except Exception as e:
            logger.error("Fatal error in SSE generator: %s", str(e))

    return Response(generate_events(), mimetype='text/event-stream')


@app.route('/api/stats')
def api_stats():
    """
    REST endpoint for aggregated statistics.

    Returns:
        JSON with global and per-camera counts
    """
    if camera_manager is None:
        return jsonify({'error': 'camera_manager not initialized'}), 500

    try:
        stats = camera_manager.get_aggregated_stats()
        return jsonify(stats), 200
    except Exception as e:
        logger.error("Error fetching stats: %s", str(e))
        return jsonify({'error': str(e)}), 500


@app.route('/api/camera/<camera_id>/stats')
def camera_stats(camera_id: str):
    """
    REST endpoint for per-camera statistics.

    Args:
        camera_id: Camera ID from config

    Returns:
        JSON with camera-specific stats or 404 if not found
    """
    if camera_manager is None:
        return jsonify({'error': 'camera_manager not initialized'}), 500

    try:
        stats = camera_manager.get_camera_stats(camera_id)
        if stats is None:
            return jsonify({'error': f'Camera {camera_id} not found'}), 404
        return jsonify(stats), 200
    except Exception as e:
        logger.error("Error fetching camera stats: %s", str(e))
        return jsonify({'error': str(e)}), 500


@app.route('/health')
def health():
    """
    Health check endpoint.
    Returns system health and device info.
    """
    if camera_manager is None:
        return jsonify({'status': 'degraded', 'error': 'camera_manager not initialized'}), 503

    try:
        stats = camera_manager.get_aggregated_stats()
        health_status = {
            'status': 'healthy' if stats['connected_count'] > 0 else 'unhealthy',
            'cameras_connected': stats['connected_count'],
            'cameras_total': stats['camera_count'],
            'inference_device': 'openvino_GPU' if camera_manager.model.device.startswith('openvino') else camera_manager.model.device,
        }
        return jsonify(health_status), 200
    except Exception as e:
        logger.error("Error in health check: %s", str(e))
        return jsonify({'status': 'error', 'error': str(e)}), 500


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({'error': 'Not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    logger.error("Internal server error: %s", str(error))
    return jsonify({'error': 'Internal server error'}), 500


def run_app(host: str = config.FLASK_HOST, port: int = config.FLASK_PORT, debug: bool = config.FLASK_DEBUG) -> None:
    """
    Run the Flask web server.

    Args:
        host: Host to bind to
        port: Port to bind to
        debug: Debug mode flag
    """
    logger.info("Starting Flask server on %s:%d", host, port)
    app.run(host=host, port=port, debug=debug, threaded=True, use_reloader=False)

