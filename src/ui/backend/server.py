"""
UI Backend WebSocket Bridge

This service bridges Redis pub/sub to WebSocket connections for the React UI.
It subscribes to the classification results channel and forwards messages to
connected web clients in real-time.
"""

import os
import json
import logging
import redis
from flask import Flask
from flask_socketio import SocketIO, emit
from flask_cors import CORS
from threading import Thread
import time

# Configure logging
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("ui-backend")

# Configuration
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_RESULTS_CHANNEL = os.getenv("REDIS_RESULTS_CHANNEL", "classification_results")
SERVER_PORT = int(os.getenv("SERVER_PORT", 5000))

# Create Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv("SECRET_KEY", "your-secret-key-here")
CORS(app)

# Create SocketIO instance
socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    async_mode='threading',
    logger=True,
    engineio_logger=False
)

# Global state
connected_clients = set()
message_count = 0
start_time = time.time()


@app.route('/health')
def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "connected_clients": len(connected_clients),
        "uptime_seconds": time.time() - start_time,
        "messages_processed": message_count
    }


@socketio.on('connect')
def handle_connect():
    """Handle client connection."""
    logger.info(f"Client connected: {request.sid if 'request' in dir() else 'unknown'}")
    connected_clients.add(request.sid if 'request' in dir() else None)
    emit('connection_status', {'status': 'connected'})


@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection."""
    logger.info(f"Client disconnected: {request.sid if 'request' in dir() else 'unknown'}")
    connected_clients.discard(request.sid if 'request' in dir() else None)


def redis_subscriber():
    """
    Subscribe to Redis pub/sub channel and forward messages to WebSocket clients.

    This runs in a separate thread and bridges Redis pub/sub to WebSocket.
    """
    global message_count

    logger.info(f"Starting Redis subscriber for channel: {REDIS_RESULTS_CHANNEL}")

    # Connect to Redis with retry logic
    max_retries = 10
    retry_delay = 2

    for attempt in range(max_retries):
        try:
            redis_client = redis.Redis(
                host=REDIS_HOST,
                port=REDIS_PORT,
                decode_responses=True
            )
            redis_client.ping()
            logger.info(f"Connected to Redis at {REDIS_HOST}:{REDIS_PORT}")
            break
        except redis.ConnectionError as e:
            logger.warning(f"Redis connection attempt {attempt + 1}/{max_retries} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                logger.error(f"Could not connect to Redis after {max_retries} attempts")
                return

    # Subscribe to channel
    pubsub = redis_client.pubsub()
    pubsub.subscribe(REDIS_RESULTS_CHANNEL)
    logger.info(f"Subscribed to Redis channel: {REDIS_RESULTS_CHANNEL}")

    # Listen for messages
    for message in pubsub.listen():
        if message['type'] == 'message':
            try:
                # Parse the classification result
                data = json.loads(message['data'])
                message_count += 1

                # Broadcast to all connected WebSocket clients
                socketio.emit('classification_result', data)

                # Log periodically
                if message_count % 100 == 0:
                    logger.info(f"Processed {message_count} messages, {len(connected_clients)} clients connected")

            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse message: {e}")
            except Exception as e:
                logger.error(f"Error processing message: {e}")


def stats_updater():
    """
    Send periodic stats updates to clients.

    This calculates and broadcasts metrics like messages per second.
    """
    global message_count, start_time
    last_count = 0
    last_time = time.time()

    while True:
        time.sleep(1)  # Update every second

        current_time = time.time()
        current_count = message_count

        # Calculate messages per second
        time_diff = current_time - last_time
        count_diff = current_count - last_count

        if time_diff > 0:
            messages_per_second = count_diff / time_diff
        else:
            messages_per_second = 0

        # Broadcast stats
        socketio.emit('stats_update', {
            'messagesPerSecond': messages_per_second,
            'totalMessages': current_count,
            'connectedClients': len(connected_clients)
        })

        last_count = current_count
        last_time = current_time


def main():
    """Main entry point."""
    logger.info("=" * 60)
    logger.info("Starting UI Backend WebSocket Bridge")
    logger.info("=" * 60)
    logger.info(f"Configuration:")
    logger.info(f"  - Redis: {REDIS_HOST}:{REDIS_PORT}")
    logger.info(f"  - Channel: {REDIS_RESULTS_CHANNEL}")
    logger.info(f"  - Server Port: {SERVER_PORT}")
    logger.info("=" * 60)

    # Start Redis subscriber in background thread
    redis_thread = Thread(target=redis_subscriber, daemon=True)
    redis_thread.start()
    logger.info("Redis subscriber thread started")

    # Start stats updater in background thread
    stats_thread = Thread(target=stats_updater, daemon=True)
    stats_thread.start()
    logger.info("Stats updater thread started")

    # Start Flask-SocketIO server
    logger.info(f"Starting WebSocket server on port {SERVER_PORT}...")
    socketio.run(
        app,
        host='0.0.0.0',
        port=SERVER_PORT,
        debug=False,
        allow_unsafe_werkzeug=True
    )


if __name__ == '__main__':
    main()
