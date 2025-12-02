"""
Unit Tests for the Traffic Generator Service

These tests verify that the generator correctly:
1. Formats messages in the expected JSON structure
2. Includes all required features
3. Handles different data types properly
4. Generates traffic in the correct modes

Run with: pytest tests/test_generator.py -v
"""

import pytest
import pandas as pd
import numpy as np
import json
import sys
import os

# Add the generator module to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'services', 'generator'))

from main import (
    record_to_message,
    generate_traffic,
    ALL_FEATURES,
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES
)


# =============================================================================
# FIXTURES - Reusable test data
# =============================================================================

@pytest.fixture
def sample_record():
    """
    Create a sample record that mimics a row from the IDS dataset.
    This is based on the actual data structure from your notebook.
    """
    data = {
        # Numerical features
        'duration': 0,
        'src_bytes': 181,
        'dst_bytes': 5450,
        'land': 0,
        'wrong_fragment': 0,
        'urgent': 0,
        'hot': 0,
        'num_failed_logins': 0,
        'logged_in': 1,
        'num_compromised': 0,
        'root_shell': 0,
        'su_attempted': 0,
        'num_root': 0,
        'num_file_creations': 0,
        'num_shells': 0,
        'num_access_files': 0,
        'num_outbound_cmds': 0,
        'is_host_login': 0,
        'is_guest_login': 0,
        'count': 8,
        'srv_count': 8,
        'serror_rate': 0.0,
        'srv_serror_rate': 0.0,
        'rerror_rate': 0.0,
        'srv_rerror_rate': 0.0,
        'same_srv_rate': 1.0,
        'diff_srv_rate': 0.0,
        'srv_diff_host_rate': 0.0,
        'dst_host_count': 9,
        'dst_host_srv_count': 9,
        'dst_host_same_srv_rate': 1.0,
        'dst_host_diff_srv_rate': 0.0,
        'dst_host_same_src_port_rate': 0.11,
        'dst_host_srv_diff_host_rate': 0.0,
        'dst_host_serror_rate': 0.0,
        'dst_host_srv_serror_rate': 0.0,
        'dst_host_rerror_rate': 0.0,
        'dst_host_srv_rerror_rate': 0.0,
        # Categorical features
        'protocol_type': 'tcp',
        'service': 'http',
        'flag': 'SF',
        # Target columns
        'target': 'normal.',
        'Attack Type': 'normal',
        'is_attack': 0
    }
    return pd.Series(data)


@pytest.fixture
def sample_attack_record():
    """
    Create a sample attack record for testing attack detection scenarios.
    """
    data = {
        'duration': 0,
        'src_bytes': 0,
        'dst_bytes': 0,
        'land': 0,
        'wrong_fragment': 0,
        'urgent': 0,
        'hot': 0,
        'num_failed_logins': 0,
        'logged_in': 0,
        'num_compromised': 0,
        'root_shell': 0,
        'su_attempted': 0,
        'num_root': 0,
        'num_file_creations': 0,
        'num_shells': 0,
        'num_access_files': 0,
        'num_outbound_cmds': 0,
        'is_host_login': 0,
        'is_guest_login': 0,
        'count': 511,
        'srv_count': 511,
        'serror_rate': 0.0,
        'srv_serror_rate': 0.0,
        'rerror_rate': 0.0,
        'srv_rerror_rate': 0.0,
        'same_srv_rate': 1.0,
        'diff_srv_rate': 0.0,
        'srv_diff_host_rate': 0.0,
        'dst_host_count': 255,
        'dst_host_srv_count': 255,
        'dst_host_same_srv_rate': 1.0,
        'dst_host_diff_srv_rate': 0.0,
        'dst_host_same_src_port_rate': 1.0,
        'dst_host_srv_diff_host_rate': 0.0,
        'dst_host_serror_rate': 0.0,
        'dst_host_srv_serror_rate': 0.0,
        'dst_host_rerror_rate': 0.0,
        'dst_host_srv_rerror_rate': 0.0,
        'protocol_type': 'icmp',
        'service': 'ecr_i',
        'flag': 'SF',
        'target': 'smurf.',
        'Attack Type': 'DoS',
        'is_attack': 1
    }
    return pd.Series(data)


@pytest.fixture
def sample_dataframe(sample_record, sample_attack_record):
    """
    Create a small DataFrame with both normal and attack records.
    """
    return pd.DataFrame([
        sample_record.to_dict(),
        sample_attack_record.to_dict(),
        sample_record.to_dict(),  # Another normal record
    ])


# =============================================================================
# MESSAGE FORMAT TESTS
# =============================================================================

class TestRecordToMessage:
    """Tests for the record_to_message function."""
    
    def test_message_has_required_fields(self, sample_record):
        """Verify that the message contains all required top-level fields."""
        message = record_to_message(sample_record, record_id=1)
        
        assert 'id' in message, "Message should have an 'id' field"
        assert 'timestamp' in message, "Message should have a 'timestamp' field"
        assert 'features' in message, "Message should have a 'features' field"
        assert 'ground_truth' in message, "Message should have a 'ground_truth' field"
        assert 'raw_log' in message, "Message should have a 'raw_log' field"
    
    def test_message_id_is_correct(self, sample_record):
        """Verify that the record ID is correctly set."""
        message = record_to_message(sample_record, record_id=42)
        assert message['id'] == 42
    
    def test_features_contains_all_required_features(self, sample_record):
        """Verify that all features expected by the classifier are present."""
        message = record_to_message(sample_record, record_id=1)
        features = message['features']
        
        for feature in ALL_FEATURES:
            assert feature in features, f"Missing feature: {feature}"
    
    def test_features_contains_correct_values(self, sample_record):
        """Verify that feature values are correctly extracted."""
        message = record_to_message(sample_record, record_id=1)
        features = message['features']
        
        assert features['duration'] == 0
        assert features['src_bytes'] == 181
        assert features['protocol_type'] == 'tcp'
        assert features['service'] == 'http'
        assert features['flag'] == 'SF'
    
    def test_ground_truth_fields(self, sample_record):
        """Verify that ground truth contains the expected labels."""
        message = record_to_message(sample_record, record_id=1)
        gt = message['ground_truth']
        
        assert gt['target'] == 'normal.'
        assert gt['attack_type'] == 'normal'
        assert gt['is_attack'] == 0
    
    def test_attack_ground_truth(self, sample_attack_record):
        """Verify that attack records have correct ground truth."""
        message = record_to_message(sample_attack_record, record_id=1)
        gt = message['ground_truth']
        
        assert gt['target'] == 'smurf.'
        assert gt['attack_type'] == 'DoS'
        assert gt['is_attack'] == 1
    
    def test_message_is_json_serializable(self, sample_record):
        """Verify that the message can be serialized to JSON without errors."""
        message = record_to_message(sample_record, record_id=1)
        
        # This should not raise an exception
        json_str = json.dumps(message)
        assert isinstance(json_str, str)
        
        # And we should be able to deserialize it back
        parsed = json.loads(json_str)
        assert parsed['id'] == message['id']
    
    def test_timestamp_is_iso_format(self, sample_record):
        """Verify that the timestamp is in ISO format."""
        message = record_to_message(sample_record, record_id=1)
        
        # ISO format should contain 'T' separator and end with timezone
        assert 'T' in message['timestamp']
        # Should be parseable
        from datetime import datetime
        # This will raise if format is invalid
        datetime.fromisoformat(message['timestamp'].replace('Z', '+00:00'))


# =============================================================================
# TRAFFIC GENERATION TESTS
# =============================================================================

class TestGenerateTraffic:
    """Tests for the traffic generation function."""
    
    def test_sequential_mode_preserves_order(self, sample_dataframe):
        """Verify that sequential mode maintains record order."""
        # We need to temporarily override LOOP to False for testing
        import main
        original_loop = main.LOOP
        main.LOOP = False
        
        try:
            generator = generate_traffic(sample_dataframe, mode="sequential")
            messages = list(generator)
            
            # Should have same number of messages as records
            assert len(messages) == len(sample_dataframe)
            
            # First message should be normal (from first record)
            assert messages[0]['ground_truth']['attack_type'] == 'normal'
            
            # Second message should be attack (from second record)
            assert messages[1]['ground_truth']['attack_type'] == 'DoS'
        finally:
            main.LOOP = original_loop
    
    def test_generator_yields_valid_messages(self, sample_dataframe):
        """Verify that the generator yields properly formatted messages."""
        import main
        original_loop = main.LOOP
        main.LOOP = False
        
        try:
            generator = generate_traffic(sample_dataframe, mode="sequential")
            
            for i, message in enumerate(generator):
                assert 'id' in message
                assert 'features' in message
                assert 'ground_truth' in message
                assert message['id'] == i + 1  # IDs should be sequential
        finally:
            main.LOOP = original_loop
    
    def test_record_ids_are_sequential(self, sample_dataframe):
        """Verify that record IDs increment properly."""
        import main
        original_loop = main.LOOP
        main.LOOP = False
        
        try:
            generator = generate_traffic(sample_dataframe, mode="sequential")
            messages = list(generator)
            
            ids = [m['id'] for m in messages]
            assert ids == [1, 2, 3]
        finally:
            main.LOOP = original_loop


# =============================================================================
# FEATURE DEFINITION TESTS
# =============================================================================

class TestFeatureDefinitions:
    """Tests to verify feature definitions match the dataset."""
    
    def test_categorical_features_count(self):
        """Verify we have the expected categorical features."""
        assert len(CATEGORICAL_FEATURES) == 3
        assert 'protocol_type' in CATEGORICAL_FEATURES
        assert 'service' in CATEGORICAL_FEATURES
        assert 'flag' in CATEGORICAL_FEATURES
    
    def test_numerical_features_count(self):
        """Verify we have the expected number of numerical features."""
        # Based on the notebook, there should be 38 numerical features
        assert len(NUMERICAL_FEATURES) == 38
    
    def test_all_features_is_union(self):
        """Verify ALL_FEATURES is the union of numerical and categorical."""
        assert len(ALL_FEATURES) == len(NUMERICAL_FEATURES) + len(CATEGORICAL_FEATURES)
        
        for f in NUMERICAL_FEATURES:
            assert f in ALL_FEATURES
        
        for f in CATEGORICAL_FEATURES:
            assert f in ALL_FEATURES


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
