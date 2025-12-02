"""
Unit Tests for the Classifier Service

These tests verify that the classifier service correctly:
1. Processes messages from the Redis queue
2. Performs inference using the model
3. Publishes results in the correct format
4. Exposes health and metrics endpoints
5. Handles errors gracefully

Run with: pytest tests/test_classifier.py -v
"""

import pytest
import json
import sys
import os
from unittest.mock import Mock, MagicMock, patch, call
from datetime import datetime, timezone

# Add the classifier module to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'services', 'classifier'))

# Mock the src.model.inference import before importing main
sys.modules['src'] = MagicMock()
sys.modules['src.model'] = MagicMock()
sys.modules['src.model.inference'] = MagicMock()

from main import (
    ServiceMetrics,
    process_message,
    publish_result,
    REDIS_QUEUE,
    REDIS_RESULTS_CHANNEL
)


# =============================================================================
# FIXTURES - Reusable test data and mocks
# =============================================================================

@pytest.fixture
def sample_traffic_message():
    """
    Create a sample traffic message as received from the generator service.
    This mimics the format produced by the traffic generator.
    """
    return {
        "id": 1,
        "timestamp": "2024-12-01T14:30:00.123456+00:00",
        "features": {
            "duration": 0,
            "protocol_type": "tcp",
            "service": "http",
            "flag": "SF",
            "src_bytes": 181,
            "dst_bytes": 5450,
            "logged_in": 1,
            "count": 8,
            "srv_count": 8,
            "serror_rate": 0.0,
            "srv_serror_rate": 0.0,
            "rerror_rate": 0.0,
            "srv_rerror_rate": 0.0,
            "same_srv_rate": 1.0,
            "diff_srv_rate": 0.0,
            "srv_diff_host_rate": 0.0,
            "dst_host_count": 9,
            "dst_host_srv_count": 9,
            "dst_host_same_srv_rate": 1.0,
            "dst_host_diff_srv_rate": 0.0,
            "dst_host_same_src_port_rate": 0.11,
            "dst_host_srv_diff_host_rate": 0.0,
            "dst_host_serror_rate": 0.0,
            "dst_host_srv_serror_rate": 0.0,
            "dst_host_rerror_rate": 0.0,
            "dst_host_srv_rerror_rate": 0.0,
        },
        "ground_truth": {
            "target": "normal.",
            "attack_type": "normal",
            "is_attack": 0
        },
        "raw_log": "0,tcp,http,SF,181,5450,..."
    }


@pytest.fixture
def sample_attack_message():
    """Create a sample attack traffic message."""
    return {
        "id": 2,
        "timestamp": "2024-12-01T14:30:01.123456+00:00",
        "features": {
            "duration": 0,
            "protocol_type": "icmp",
            "service": "ecr_i",
            "flag": "SF",
            "src_bytes": 0,
            "dst_bytes": 0,
            "logged_in": 0,
            "count": 511,
            "srv_count": 511,
            "serror_rate": 0.0,
            "srv_serror_rate": 0.0,
            "rerror_rate": 0.0,
            "srv_rerror_rate": 0.0,
            "same_srv_rate": 1.0,
            "diff_srv_rate": 0.0,
            "srv_diff_host_rate": 0.0,
            "dst_host_count": 255,
            "dst_host_srv_count": 255,
            "dst_host_same_srv_rate": 1.0,
            "dst_host_diff_srv_rate": 0.0,
            "dst_host_same_src_port_rate": 1.0,
            "dst_host_srv_diff_host_rate": 0.0,
            "dst_host_serror_rate": 0.0,
            "dst_host_srv_serror_rate": 0.0,
            "dst_host_rerror_rate": 0.0,
            "dst_host_srv_rerror_rate": 0.0,
        },
        "ground_truth": {
            "target": "smurf.",
            "attack_type": "DoS",
            "is_attack": 1
        },
        "raw_log": "0,icmp,ecr_i,SF,0,0,..."
    }


@pytest.fixture
def mock_classifier():
    """Create a mock classifier that returns predictable results."""
    classifier = Mock()

    # Normal prediction result
    classifier.predict_one.return_value = {
        "prediction": "normal",
        "prediction_label": 0,
        "confidence": 0.92,
        "probabilities": {
            "normal": 0.92,
            "attack": 0.08
        }
    }

    return classifier


@pytest.fixture
def mock_attack_classifier():
    """Create a mock classifier that predicts attacks."""
    classifier = Mock()

    # Attack prediction result
    classifier.predict_one.return_value = {
        "prediction": "attack",
        "prediction_label": 1,
        "confidence": 0.87,
        "probabilities": {
            "normal": 0.13,
            "attack": 0.87
        }
    }

    return classifier


@pytest.fixture
def mock_redis():
    """Create a mock Redis client."""
    redis_client = Mock()
    redis_client.ping.return_value = True
    redis_client.publish.return_value = 1  # Number of subscribers
    return redis_client


# =============================================================================
# SERVICE METRICS TESTS
# =============================================================================

class TestServiceMetrics:
    """Tests for the ServiceMetrics class."""

    def test_initial_state(self):
        """Verify that metrics start in a clean state."""
        metrics = ServiceMetrics()

        assert metrics.messages_processed == 0
        assert metrics.predictions_normal == 0
        assert metrics.predictions_attack == 0
        assert metrics.errors == 0
        assert metrics.last_prediction_time is None
        assert metrics.model_loaded is False
        assert metrics.redis_connected is False

    def test_record_normal_prediction(self):
        """Verify that normal predictions are counted correctly."""
        metrics = ServiceMetrics()

        metrics.record_prediction("normal")

        assert metrics.messages_processed == 1
        assert metrics.predictions_normal == 1
        assert metrics.predictions_attack == 0
        assert metrics.last_prediction_time is not None

    def test_record_attack_prediction(self):
        """Verify that attack predictions are counted correctly."""
        metrics = ServiceMetrics()

        metrics.record_prediction("attack")

        assert metrics.messages_processed == 1
        assert metrics.predictions_normal == 0
        assert metrics.predictions_attack == 1

    def test_record_multiple_predictions(self):
        """Verify that multiple predictions are counted correctly."""
        metrics = ServiceMetrics()

        metrics.record_prediction("normal")
        metrics.record_prediction("attack")
        metrics.record_prediction("normal")
        metrics.record_prediction("attack")

        assert metrics.messages_processed == 4
        assert metrics.predictions_normal == 2
        assert metrics.predictions_attack == 2

    def test_record_error(self):
        """Verify that errors are counted."""
        metrics = ServiceMetrics()

        metrics.record_error()
        metrics.record_error()

        assert metrics.errors == 2

    def test_get_uptime(self):
        """Verify that uptime is tracked."""
        import time
        metrics = ServiceMetrics()

        time.sleep(0.1)  # Sleep for 100ms
        uptime = metrics.get_uptime()

        assert uptime >= 0.1
        assert uptime < 1.0  # Should be less than 1 second

    def test_get_metrics_structure(self):
        """Verify that get_metrics returns the expected structure."""
        metrics = ServiceMetrics()
        metrics.model_loaded = True
        metrics.redis_connected = True
        metrics.record_prediction("normal")
        metrics.record_prediction("attack")

        result = metrics.get_metrics()

        assert 'uptime_seconds' in result
        assert 'messages_processed' in result
        assert 'predictions' in result
        assert 'errors' in result
        assert 'last_prediction_time' in result
        assert 'status' in result

        assert result['messages_processed'] == 2
        assert result['predictions']['normal'] == 1
        assert result['predictions']['attack'] == 1
        assert result['status']['model_loaded'] is True
        assert result['status']['redis_connected'] is True

    def test_attack_rate_calculation(self):
        """Verify that attack rate is calculated correctly."""
        metrics = ServiceMetrics()

        # Record 7 normal, 3 attack = 30% attack rate
        for _ in range(7):
            metrics.record_prediction("normal")
        for _ in range(3):
            metrics.record_prediction("attack")

        result = metrics.get_metrics()
        attack_rate = result['predictions']['attack_rate']

        assert attack_rate == 0.3  # 3/10 = 0.3

    def test_attack_rate_with_no_predictions(self):
        """Verify that attack rate is 0 when no predictions made."""
        metrics = ServiceMetrics()

        result = metrics.get_metrics()
        attack_rate = result['predictions']['attack_rate']

        assert attack_rate == 0.0


# =============================================================================
# MESSAGE PROCESSING TESTS
# =============================================================================

class TestProcessMessage:
    """Tests for the process_message function."""

    def test_process_normal_message(self, sample_traffic_message, mock_classifier):
        """Verify that normal traffic is processed correctly."""
        result = process_message(mock_classifier, sample_traffic_message)

        # Should call predict_one with features
        mock_classifier.predict_one.assert_called_once_with(
            sample_traffic_message['features']
        )

        # Result should have required fields
        assert 'id' in result
        assert 'timestamp' in result
        assert 'original_log' in result
        assert 'prediction' in result
        assert 'score' in result
        assert 'details' in result

        # Should predict "normal" (not "suspicious")
        assert result['prediction'] == 'normal'
        assert result['id'] == sample_traffic_message['id']

    def test_process_attack_message(self, sample_attack_message, mock_attack_classifier):
        """Verify that attack traffic is processed correctly."""
        result = process_message(mock_attack_classifier, sample_attack_message)

        # Should predict "suspicious"
        assert result['prediction'] == 'suspicious'
        assert result['score'] == 0.87

    def test_result_format_compliance(self, sample_traffic_message, mock_classifier):
        """Verify that result format matches CLAUDE.md specification."""
        result = process_message(mock_classifier, sample_traffic_message)

        # Per CLAUDE.md, result should have:
        # - original_log
        # - prediction: "suspicious" | "normal"
        # - score: confidence score
        # - timestamp

        assert result['original_log'] == sample_traffic_message
        assert result['prediction'] in ['suspicious', 'normal']
        assert isinstance(result['score'], (int, float))
        assert 'T' in result['timestamp']  # ISO format

    def test_result_includes_details(self, sample_traffic_message, mock_classifier):
        """Verify that result includes detailed prediction info."""
        result = process_message(mock_classifier, sample_traffic_message)

        assert 'details' in result
        assert 'prediction_label' in result['details']
        assert 'probabilities' in result['details']

    def test_result_is_json_serializable(self, sample_traffic_message, mock_classifier):
        """Verify that result can be serialized to JSON."""
        result = process_message(mock_classifier, sample_traffic_message)

        # Should not raise an exception
        json_str = json.dumps(result)
        assert isinstance(json_str, str)

        # Should be deserializable
        parsed = json.loads(json_str)
        assert parsed['id'] == result['id']

    def test_message_without_features(self, mock_classifier):
        """Verify that messages without features are handled."""
        message = {"id": 1, "timestamp": "2024-12-01T14:30:00Z"}

        result = process_message(mock_classifier, message)

        # Should return None for invalid messages
        assert result is None

        # Should NOT call the classifier
        mock_classifier.predict_one.assert_not_called()

    def test_empty_features(self, mock_classifier):
        """Verify that empty features dict is handled."""
        message = {"id": 1, "features": {}}

        result = process_message(mock_classifier, message)

        assert result is None
        mock_classifier.predict_one.assert_not_called()

    def test_classifier_raises_exception(self, sample_traffic_message):
        """Verify that classifier exceptions are handled gracefully."""
        classifier = Mock()
        classifier.predict_one.side_effect = Exception("Model error")

        result = process_message(classifier, sample_traffic_message)

        # Should return None on error
        assert result is None


# =============================================================================
# RESULT PUBLISHING TESTS
# =============================================================================

class TestPublishResult:
    """Tests for the publish_result function."""

    def test_publish_to_correct_channel(self, mock_redis):
        """Verify that results are published to the correct Redis channel."""
        result = {
            "id": 1,
            "prediction": "normal",
            "score": 0.92
        }

        publish_result(mock_redis, result)

        # Should call publish with correct channel
        mock_redis.publish.assert_called_once()
        call_args = mock_redis.publish.call_args

        assert call_args[0][0] == REDIS_RESULTS_CHANNEL

    def test_publish_serializes_to_json(self, mock_redis):
        """Verify that result is serialized to JSON before publishing."""
        result = {
            "id": 1,
            "prediction": "normal",
            "score": 0.92
        }

        publish_result(mock_redis, result)

        # Get the published message
        call_args = mock_redis.publish.call_args
        published_message = call_args[0][1]

        # Should be valid JSON
        parsed = json.loads(published_message)
        assert parsed['id'] == 1
        assert parsed['prediction'] == 'normal'

    def test_publish_handles_redis_error(self, mock_redis):
        """Verify that Redis publish errors are handled."""
        mock_redis.publish.side_effect = Exception("Redis error")

        result = {"id": 1, "prediction": "normal"}

        # Should not raise exception
        publish_result(mock_redis, result)


# =============================================================================
# API ENDPOINT TESTS
# =============================================================================

class TestAPIEndpoints:
    """Tests for the FastAPI health and metrics endpoints."""

    def test_health_endpoint_healthy(self):
        """Verify that health endpoint returns healthy status."""
        from main import app, metrics
        from fastapi.testclient import TestClient

        # Set metrics to healthy state
        metrics.model_loaded = True
        metrics.redis_connected = True

        client = TestClient(app)
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()

        assert data['status'] == 'healthy'
        assert data['model_loaded'] is True
        assert data['redis_connected'] is True
        assert 'uptime_seconds' in data

    def test_health_endpoint_unhealthy_no_model(self):
        """Verify that health endpoint returns unhealthy when model not loaded."""
        from main import app, metrics
        from fastapi.testclient import TestClient

        # Set metrics to unhealthy state
        metrics.model_loaded = False
        metrics.redis_connected = True

        client = TestClient(app)
        response = client.get("/health")

        assert response.status_code == 503
        data = response.json()

        assert data['status'] == 'unhealthy'
        assert data['model_loaded'] is False

    def test_health_endpoint_unhealthy_no_redis(self):
        """Verify that health endpoint returns unhealthy when Redis not connected."""
        from main import app, metrics
        from fastapi.testclient import TestClient

        # Set metrics to unhealthy state
        metrics.model_loaded = True
        metrics.redis_connected = False

        client = TestClient(app)
        response = client.get("/health")

        assert response.status_code == 503
        data = response.json()

        assert data['status'] == 'unhealthy'
        assert data['redis_connected'] is False

    def test_metrics_endpoint(self):
        """Verify that metrics endpoint returns detailed metrics."""
        from main import app, metrics
        from fastapi.testclient import TestClient

        # Reset metrics to clean state
        metrics.messages_processed = 0
        metrics.predictions_normal = 0
        metrics.predictions_attack = 0
        metrics.errors = 0

        # Set some metrics
        metrics.model_loaded = True
        metrics.redis_connected = True
        metrics.record_prediction("normal")
        metrics.record_prediction("attack")

        client = TestClient(app)
        response = client.get("/metrics")

        assert response.status_code == 200
        data = response.json()

        assert 'uptime_seconds' in data
        assert 'messages_processed' in data
        assert 'predictions' in data
        assert data['messages_processed'] == 2
        assert data['predictions']['normal'] == 1
        assert data['predictions']['attack'] == 1

    def test_root_endpoint(self):
        """Verify that root endpoint returns service info."""
        from main import app
        from fastapi.testclient import TestClient

        client = TestClient(app)
        response = client.get("/")

        assert response.status_code == 200
        data = response.json()

        assert 'service' in data
        assert 'version' in data
        assert 'endpoints' in data
        assert data['service'] == 'IDS Classifier'


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegration:
    """Integration tests for the full message processing flow."""

    def test_full_processing_flow(self, sample_traffic_message, mock_classifier, mock_redis):
        """Test the complete flow from message to published result."""
        # Process the message
        result = process_message(mock_classifier, sample_traffic_message)

        assert result is not None

        # Publish the result
        publish_result(mock_redis, result)

        # Verify classifier was called
        mock_classifier.predict_one.assert_called_once()

        # Verify Redis publish was called
        mock_redis.publish.assert_called_once()

        # Verify published message is valid JSON
        published_data = mock_redis.publish.call_args[0][1]
        parsed = json.loads(published_data)

        assert parsed['id'] == sample_traffic_message['id']
        assert parsed['prediction'] in ['normal', 'suspicious']

    def test_batch_processing(self, sample_traffic_message, sample_attack_message,
                            mock_classifier, mock_attack_classifier, mock_redis):
        """Test processing multiple messages in sequence."""
        # Process normal message
        result1 = process_message(mock_classifier, sample_traffic_message)
        publish_result(mock_redis, result1)

        # Process attack message
        result2 = process_message(mock_attack_classifier, sample_attack_message)
        publish_result(mock_redis, result2)

        # Should have published twice
        assert mock_redis.publish.call_count == 2

        # First result should be normal
        assert result1['prediction'] == 'normal'

        # Second result should be suspicious
        assert result2['prediction'] == 'suspicious'


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
