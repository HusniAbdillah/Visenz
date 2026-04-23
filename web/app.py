"""
Flask web server with Server-Sent Events (SSE) and Admin APIs for edge-vision-counter V2.
Provides REST endpoints for system health, aggregated stats, calibration, and reset.
Multi-camera dashboard support with real-time SSE updates.
"""

import atexit
import csv
import json
import logging
import importlib
import io
import socket
import time
from datetime import datetime
from typing import Iterator, Optional
from flask import Flask, render_template, Response, jsonify, request
import pytz

import config
from core.state_manager import StateManager

logger = logging.getLogger(__name__)
WIB_TZ = pytz.timezone("Asia/Jakarta")

app = Flask(__name__, template_folder='templates')
app.config['JSON_SORT_KEYS'] = False

camera_manager = None
state_manager: Optional[StateManager] = None
_mdns_zeroconf = None
_mdns_service_info = None


def _wib_now() -> str:
    return datetime.now(WIB_TZ).strftime("%Y-%m-%d %H:%M:%S")


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


def _resolve_host_ip() -> str:
    """Resolve the primary non-loopback IPv4 address for mDNS advertisement."""
    try:
        host_name = socket.gethostname()
        candidates = socket.gethostbyname_ex(host_name)[2]
        for candidate in candidates:
            if not candidate.startswith("127."):
                return candidate
    except Exception:
        pass

    return "127.0.0.1"


def start_mdns_service(port: int = config.FLASK_PORT) -> bool:
    """Advertise the dashboard as counter-gww.local over mDNS when available."""
    global _mdns_zeroconf, _mdns_service_info

    if not getattr(config, "MDNS_ENABLED", True):
        return False

    if _mdns_zeroconf is not None:
        return True

    try:
        zeroconf_module = importlib.import_module("zeroconf")
        Zeroconf = getattr(zeroconf_module, "Zeroconf")
        ServiceInfo = getattr(zeroconf_module, "ServiceInfo")
        host_ip = _resolve_host_ip()
        service_name = f"{config.MDNS_SERVICE_NAME}._http._tcp.local."
        host_name = f"{config.MDNS_HOSTNAME}.local."
        info = ServiceInfo(
            type_="_http._tcp.local.",
            name=service_name,
            addresses=[socket.inet_aton(host_ip)],
            port=port,
            properties={"path": "/"},
            server=host_name,
        )
        zeroconf = Zeroconf()
        zeroconf.register_service(info)
        _mdns_zeroconf = zeroconf
        _mdns_service_info = info
        atexit.register(stop_mdns_service)
        logger.info("mDNS service registered as http://%s.local:%d", config.MDNS_HOSTNAME, port)
        return True
    except Exception as exc:
        logger.warning("Failed to register mDNS service: %s", str(exc))
        return False


def stop_mdns_service() -> None:
    """Unregister the mDNS service if it was started."""
    global _mdns_zeroconf, _mdns_service_info

    if _mdns_zeroconf is None or _mdns_service_info is None:
        return

    try:
        _mdns_zeroconf.unregister_service(_mdns_service_info)
        _mdns_zeroconf.close()
    except Exception as exc:
        logger.debug("Error stopping mDNS service: %s", str(exc))
    finally:
        _mdns_zeroconf = None
        _mdns_service_info = None


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
                    if state_manager is None:
                        logger.error("StateManager instance is None in event generator")
                        yield f"data: {json.dumps({'error': 'state_manager not initialized'})}\n\n"
                        time.sleep(1)
                        continue

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
                        'session_id': stats.get('session_id'),
                        'timezone': stats.get('timezone', 'Asia/Jakarta'),
                        'last_reset_at': stats.get('last_reset_at'),
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
        analytics = state_manager.get_analytics(interval_minutes=int(request.args.get('interval_minutes', 15)))
        
        result = {
            'total_in': stats['total_in'],
            'total_out': stats['total_out'],
            'current_inside': stats['current_inside'],
            'session_id': stats.get('session_id'),
            'timezone': stats.get('timezone', 'Asia/Jakarta'),
            'last_reset_at': stats.get('last_reset_at'),
            'interval_minutes': analytics.get('interval_minutes', 15),
            'time_series': analytics.get('time_series', []),
            'camera_summary': analytics.get('camera_summary', []),
            'generated_at': _wib_now(),
        }

        if camera_manager is not None:
            try:
                aggregated = camera_manager.get_aggregated_stats()
                result['camera_count'] = aggregated.get('camera_count', 0)
                result['connected_count'] = aggregated.get('connected_count', 0)
                result['cameras'] = aggregated.get('cameras', [])
            except Exception as e:
                logger.warning("Error getting camera stats: %s", str(e))

        if camera_manager is not None and getattr(camera_manager, 'model', None) is not None:
            try:
                result['gpu'] = camera_manager.model.get_gpu_stats()
            except Exception as e:
                logger.warning("Error getting GPU stats: %s", str(e))

        result['service_url'] = f"http://{config.MDNS_HOSTNAME}.local:{config.FLASK_PORT}"

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
        data = request.get_json(silent=True) or {}
        provided_password = str(data.get('password', ''))
        reset_password = str(getattr(config, 'ADMIN_RESET_PASSWORD', 'admin'))

        if provided_password != reset_password:
            return jsonify({'error': 'Invalid password'}), 403

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


@app.route('/api/export')
def export_logs():
    """Export SQLite event logs as CSV for active session by default."""
    if state_manager is None:
        return jsonify({'error': 'state_manager not initialized'}), 500

    try:
        export_all = str(request.args.get('all', '0')).lower() in {'1', 'true', 'yes'}
        payload = state_manager.get_event_logs(only_current_session=not export_all)
        session_id = payload.get('session_id', 'unknown')
        rows = payload.get('rows', [])

        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(['id', 'timestamp_wib', 'camera_id', 'direction', 'session_id'])
        for row in rows:
            writer.writerow([
                row.get('id'),
                row.get('timestamp_wib'),
                row.get('camera_id'),
                row.get('direction'),
                row.get('session_id'),
            ])

        csv_text = output.getvalue()
        output.close()
        filename = f"edge_vision_logs_{session_id}.csv"

        return Response(
            csv_text,
            mimetype='text/csv',
            headers={
                'Content-Disposition': f'attachment; filename={filename}',
                'Cache-Control': 'no-store',
            },
        )
    except Exception as e:
        logger.error('Error exporting CSV: %s', str(e))
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
    start_mdns_service(port)
    logger.info("Starting Flask server on %s:%d", host, port)
    app.run(
        host=host,
        port=port,
        debug=debug,
        threaded=True,
        use_reloader=False
    )
