"""
Flask web server with Server-Sent Events (SSE) and Admin APIs for edge-vision-counter V2.
Provides REST endpoints for system health, aggregated stats, calibration, and reset.
Multi-camera dashboard support with real-time SSE updates.
"""

import logging
import json
import time
from typing import Iterator, Optional
from flask import Flask, render_template, Response, jsonify, request

import config
from core.state_manager import StateManager

logger = logging.getLogger(__name__)

app = Flask(__name__, template_folder='templates')
app.config['JSON_SORT_KEYS'] = False

camera_manager = None
state_manager: Optional[StateManager] = None


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


def set_state_manager(manager: StateManager) -> None:
    """
    Inject the StateManager instance into the Flask app.
    Called from main.py during initialization.

    Args:
        manager: StateManager instance
    """
    global state_manager
    state_manager = manager
    logger.info("StateManager instance set in Flask app")


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
        SSE formatted data string with JSON payload every 0.5 seconds.
    """
    if state_manager is None:
        logger.error("StateManager instance not initialized")
        return Response(
            "error: state_manager not initialized",
            status=500,
            mimetype='text/plain'
        )

    def generate_events() -> Iterator[str]:
        """Generate SSE events with aggregated count updates."""
        try:
            logger.info("SSE client connected")
            last_data = None

            while True:
                try:
                    stats = state_manager.get_stats()

                    camera_count = 0
                    connected_count = 0
                    
                    if camera_manager is not None:
                        try:
                            aggregated = camera_manager.get_aggregated_stats()
                            camera_count = aggregated.get('camera_count', 0)
                            connected_count = aggregated.get('connected_count', 0)
                        except Exception:
                            pass

                    data = {
                        'total_in': stats['total_in'],
                        'total_out': stats['total_out'],
                        'current_inside': stats['current_inside'],
                        'camera_count': camera_count,
                        'connected_count': connected_count,
                        'timestamp': time.time(),
                    }

                    if data != last_data:
                        yield f"data: {json.dumps(data)}\n\n"
                        last_data = data.copy()

                    time.sleep(config.SSE_UPDATE_INTERVAL)

                except Exception as e:
                    logger.error("Error in SSE stream: %s", str(e))
                    yield f"data: {json.dumps({'error': str(e)})}\n\n"
                    time.sleep(1)

        except GeneratorExit:
            logger.info("SSE client disconnected")
        except Exception as e:
            logger.error("Fatal error in SSE generator: %s", str(e))

    return Response(
        generate_events(),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'X-Accel-Buffering': 'no',
        }
    )


@app.route('/api/stats')
def api_stats():
    """
    REST endpoint for aggregated statistics.

    Returns:
        JSON with global and per-camera counts.
    """
    if state_manager is None:
        return jsonify({'error': 'state_manager not initialized'}), 500

    try:
        stats = state_manager.get_stats()
        
        result = {
            'total_in': stats['total_in'],
            'total_out': stats['total_out'],
            'current_inside': stats['current_inside'],
        }

        if camera_manager is not None:
            try:
                aggregated = camera_manager.get_aggregated_stats()
                result['camera_count'] = aggregated.get('camera_count', 0)
                result['connected_count'] = aggregated.get('connected_count', 0)
                result['cameras'] = aggregated.get('cameras', [])
            except Exception as e:
                logger.warning("Error getting camera stats: %s", str(e))

        return jsonify(result), 200

    except Exception as e:
        logger.error("Error fetching stats: %s", str(e))
        return jsonify({'error': str(e)}), 500


@app.route('/api/camera/<camera_id>/stats')
def camera_stats(camera_id: str):
    """
    REST endpoint for per-camera statistics.

    Args:
        camera_id: Camera ID from config.

    Returns:
        JSON with camera-specific stats or 404 if not found.
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


@app.route('/api/set_count', methods=['POST'])
def set_count():
    """
    Admin API: Adjust total_out based on total_in so that
    (total_in - total_out) equals the target_inside value.

    Request JSON:
        {"target_inside": <int>}

    Returns:
        JSON with success status and updated stats.
    """
    if state_manager is None:
        return jsonify({'error': 'state_manager not initialized'}), 500

    try:
        data = request.get_json()
        
        if data is None:
            return jsonify({'error': 'Invalid JSON body'}), 400

        target_inside = data.get('target_inside')
        
        if target_inside is None:
            return jsonify({'error': 'target_inside is required'}), 400

        try:
            target_inside = int(target_inside)
        except (ValueError, TypeError):
            return jsonify({'error': 'target_inside must be an integer'}), 400

        if target_inside < 0:
            return jsonify({'error': 'target_inside cannot be negative'}), 400

        state_manager.set_current_inside(target_inside)
        
        updated_stats = state_manager.get_stats()
        
        logger.info(
            "Manual calibration via API: target_inside=%d, new stats=%s",
            target_inside,
            updated_stats
        )

        return jsonify({
            'success': True,
            'message': f'Current inside count set to {target_inside}',
            'stats': updated_stats,
        }), 200

    except Exception as e:
        logger.error("Error in set_count API: %s", str(e))
        return jsonify({'error': str(e)}), 500


@app.route('/api/reset', methods=['POST'])
def reset():
    """
    Admin API: Reset all counts to zero.

    Returns:
        JSON with success status.
    """
    if state_manager is None:
        return jsonify({'error': 'state_manager not initialized'}), 500

    try:
        state_manager.reset()
        
        logger.info("All counts reset via API")

        return jsonify({
            'success': True,
            'message': 'All counts have been reset to zero',
            'stats': state_manager.get_stats(),
        }), 200

    except Exception as e:
        logger.error("Error in reset API: %s", str(e))
        return jsonify({'error': str(e)}), 500


@app.route('/health')
def health():
    """
    Health check endpoint.
    Returns system health and device info.
    """
    health_data = {
        'status': 'healthy',
        'state_manager': state_manager is not None,
        'camera_manager': camera_manager is not None,
    }

    if camera_manager is not None:
        try:
            stats = camera_manager.get_aggregated_stats()
            health_data['cameras_connected'] = stats['connected_count']
            health_data['cameras_total'] = stats['camera_count']
            health_data['inference_device'] = camera_manager.model.get_device()
            
            if stats['connected_count'] == 0 and stats['camera_count'] > 0:
                health_data['status'] = 'degraded'
        except Exception as e:
            logger.error("Error in health check: %s", str(e))
            health_data['status'] = 'degraded'
            health_data['error'] = str(e)

    status_code = 200 if health_data['status'] == 'healthy' else 503
    return jsonify(health_data), status_code


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({'error': 'Not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    logger.error("Internal server error: %s", str(error))
    return jsonify({'error': 'Internal server error'}), 500


def run_app(
    host: str = config.FLASK_HOST,
    port: int = config.FLASK_PORT,
    debug: bool = config.FLASK_DEBUG
) -> None:
    """
    Run the Flask web server.

    Args:
        host: Host to bind to.
        port: Port to bind to.
        debug: Debug mode flag.
    """
    logger.info("Starting Flask server on %s:%d", host, port)
    app.run(
        host=host,
        port=port,
        debug=debug,
        threaded=True,
        use_reloader=False
    )
