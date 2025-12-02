"""
Traffic Generator Service for the Mini ML-based IDS Platform

This service simulates real-time network traffic by reading records from the test
dataset and publishing them to a Redis queue. The classifier service will then
consume these messages and perform inference.

The generator acts as a "replay machine" - it takes historical network logs and
streams them as if they were happening live, which is perfect for testing and
demonstrating the IDS without needing actual network capture tools.
"""

import os
import json
import time
import logging
import redis
import pandas as pd
import kagglehub
from datetime import datetime, timezone
from typing import Iterator, Dict, Any

# Configure logging so we can see what the generator is doing
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("traffic-generator")


# =============================================================================
# CONFIGURATION
# =============================================================================
# These are the settings that control how the generator behaves.
# They're read from environment variables so you can easily change them
# in docker-compose.yml without modifying the code.

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_QUEUE = os.getenv("REDIS_QUEUE", "traffic_queue")

# TRAFFIC_RATE controls how many messages per second the generator produces.
# A higher rate means more load on the classifier. Start low for testing!
TRAFFIC_RATE = float(os.getenv("TRAFFIC_RATE", 10))

# MODE can be "sequential" (replay in order) or "random" (random sampling)
# Sequential is useful for reproducing specific scenarios
# Random is more realistic for general testing
MODE = os.getenv("GENERATOR_MODE", "sequential")

# Whether to loop forever or stop after one pass through the dataset
LOOP = os.getenv("GENERATOR_LOOP", "true").lower() == "true"


# =============================================================================
# FEATURE DEFINITIONS
# =============================================================================
# These must match exactly what your model expects!
# I extracted these from your Jupyter notebook.

CATEGORICAL_FEATURES = ['protocol_type', 'service', 'flag']

NUMERICAL_FEATURES = [
    'duration', 'src_bytes', 'dst_bytes', 'land', 'wrong_fragment',
    'urgent', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised',
    'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
    'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
    'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
    'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
    'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
    'dst_host_serror_rate', 'dst_host_srv_serror_rate',
    'dst_host_rerror_rate', 'dst_host_srv_rerror_rate'
]

ALL_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES


# =============================================================================
# DATA LOADING
# =============================================================================

def load_dataset() -> pd.DataFrame:
    """
    Load the IDS dataset from Kaggle.
    
    This function downloads the dataset if it's not already cached locally.
    In a production scenario, you'd probably load from a local file or database,
    but for this project we're using the same source as your training notebook.
    """
    logger.info("Loading dataset from Kaggle...")
    
    # Check if we have a local data path (for Docker deployment)
    local_data_path = os.getenv("DATA_PATH")
    
    if local_data_path and os.path.exists(local_data_path):
        logger.info(f"Loading from local path: {local_data_path}")
        data = pd.read_csv(local_data_path)
    else:
        # Download from Kaggle (same as your notebook)
        logger.info("Downloading from Kaggle...")
        path = kagglehub.dataset_download("ayushparwal2026/intrusion-detection-classifier")
        data = pd.read_csv(path + '/datacopy.csv')
    
    logger.info(f"Loaded {len(data)} records")
    return data


def prepare_test_data(data: pd.DataFrame, test_fraction: float = 0.15) -> pd.DataFrame:
    """
    Extract the test portion of the dataset.
    
    We use the same split ratio as your notebook (70/15/15) and the same
    random seed to ensure we're testing on the exact same data that wasn't
    used for training. This is important for honest evaluation!
    
    Args:
        data: The full dataset
        test_fraction: Fraction of data to use for testing (default 15%)
    
    Returns:
        DataFrame containing only the test records
    """
    # Create the is_attack column (same as your notebook)
    data = data.copy()
    data['is_attack'] = (data['Attack Type'] != 'normal').astype(int)
    
    # We need to replicate your exact split to get the same test set
    # Your notebook does: 70% train, then 50% of remaining (15% val, 15% test)
    from sklearn.model_selection import train_test_split
    
    # First split: 70% train, 30% temp
    _, temp_data = train_test_split(
        data,
        test_size=0.30,
        random_state=42,
        stratify=data['is_attack']
    )
    
    # Second split: 50% validation, 50% test (of the 30%)
    _, test_data = train_test_split(
        temp_data,
        test_size=0.50,
        random_state=42,
        stratify=temp_data['is_attack']
    )
    
    logger.info(f"Test set: {len(test_data)} records")
    logger.info(f"  - Normal traffic: {len(test_data[test_data['is_attack'] == 0])}")
    logger.info(f"  - Attack traffic: {len(test_data[test_data['is_attack'] == 1])}")
    
    return test_data


# =============================================================================
# MESSAGE FORMATTING
# =============================================================================

def record_to_message(record: pd.Series, record_id: int) -> Dict[str, Any]:
    """
    Convert a single dataset record to the JSON message format expected
    by the classifier service.
    
    The message format is designed to:
    1. Carry all features needed for classification
    2. Include metadata (timestamp, ID) for tracking
    3. Preserve ground truth for evaluation
    4. Keep the raw data for debugging/display
    
    Args:
        record: A single row from the dataset
        record_id: Unique identifier for this message
    
    Returns:
        Dictionary ready to be JSON-serialized and sent to Redis
    """
    # Extract features as a dictionary
    # We convert numpy types to Python native types for JSON serialization
    features = {}
    for feature in ALL_FEATURES:
        value = record[feature]
        # Handle numpy types that don't serialize to JSON well
        if pd.isna(value):
            features[feature] = None
        elif hasattr(value, 'item'):  # numpy scalar
            features[feature] = value.item()
        else:
            features[feature] = value
    
    # Build the complete message
    message = {
        "id": record_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "features": features,
        # Ground truth - useful for evaluating classifier performance
        "ground_truth": {
            "target": record['target'],
            "attack_type": record['Attack Type'],
            "is_attack": int(record['is_attack'])
        },
        # Raw log as CSV string for display purposes
        "raw_log": ",".join(str(record[f]) for f in ALL_FEATURES)
    }
    
    return message


# =============================================================================
# TRAFFIC GENERATION
# =============================================================================

def generate_traffic(test_data: pd.DataFrame, mode: str = "sequential") -> Iterator[Dict[str, Any]]:
    """
    Generator function that yields traffic messages one at a time.
    
    Using a generator (with yield) is memory-efficient because we don't need
    to create all messages at once. This is important when dealing with
    large datasets!
    
    Args:
        test_data: The test dataset to stream
        mode: "sequential" or "random"
    
    Yields:
        Message dictionaries ready to be sent to Redis
    """
    record_id = 0
    
    while True:  # Loop forever if LOOP is True
        if mode == "random":
            # Random sampling - shuffle the dataframe
            shuffled_data = test_data.sample(frac=1).reset_index(drop=True)
        else:
            # Sequential - maintain original order
            shuffled_data = test_data.reset_index(drop=True)
        
        # Iterate through records
        for idx, record in shuffled_data.iterrows():
            record_id += 1
            yield record_to_message(record, record_id)
        
        if not LOOP:
            logger.info("Completed one pass through the dataset. Stopping.")
            break
        else:
            logger.info("Completed one pass through the dataset. Looping...")


# =============================================================================
# REDIS PUBLISHING
# =============================================================================

def create_redis_connection() -> redis.Redis:
    """
    Create and test a connection to Redis.
    
    We use a retry loop because when running in Docker, Redis might not be
    ready immediately when our service starts.
    """
    max_retries = 10
    retry_delay = 2  # seconds
    
    for attempt in range(max_retries):
        try:
            client = redis.Redis(
                host=REDIS_HOST,
                port=REDIS_PORT,
                decode_responses=True  # Return strings instead of bytes
            )
            # Test the connection
            client.ping()
            logger.info(f"Connected to Redis at {REDIS_HOST}:{REDIS_PORT}")
            return client
        except redis.ConnectionError as e:
            logger.warning(f"Redis connection attempt {attempt + 1}/{max_retries} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                raise RuntimeError(f"Could not connect to Redis after {max_retries} attempts")


def publish_traffic(redis_client: redis.Redis, traffic_generator: Iterator[Dict[str, Any]]):
    """
    Main loop that publishes traffic messages to Redis.
    
    This function controls the rate at which messages are sent using a simple
    sleep mechanism. The delay between messages is calculated from TRAFFIC_RATE.
    
    Args:
        redis_client: Connected Redis client
        traffic_generator: Generator yielding message dictionaries
    """
    # Calculate delay between messages
    # If TRAFFIC_RATE is 10, we want 0.1 seconds between messages
    delay = 1.0 / TRAFFIC_RATE
    logger.info(f"Publishing at {TRAFFIC_RATE} messages/second (delay: {delay:.3f}s)")
    
    message_count = 0
    start_time = time.time()
    
    try:
        for message in traffic_generator:
            # Serialize to JSON and push to Redis list
            # RPUSH adds to the right (end) of the list
            # The classifier will use BLPOP to take from the left (front)
            # This creates a FIFO (First-In-First-Out) queue
            redis_client.rpush(REDIS_QUEUE, json.dumps(message))
            
            message_count += 1
            
            # Log progress periodically
            if message_count % 100 == 0:
                elapsed = time.time() - start_time
                actual_rate = message_count / elapsed
                logger.info(
                    f"Published {message_count} messages | "
                    f"Rate: {actual_rate:.1f} msg/s | "
                    f"Queue length: {redis_client.llen(REDIS_QUEUE)}"
                )
            
            # Sleep to control the rate
            time.sleep(delay)
            
    except KeyboardInterrupt:
        logger.info("Received shutdown signal")
    finally:
        elapsed = time.time() - start_time
        logger.info(f"Published {message_count} messages in {elapsed:.1f} seconds")


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    """
    Main entry point for the traffic generator service.
    
    This orchestrates the entire flow:
    1. Load and prepare the dataset
    2. Connect to Redis
    3. Start publishing traffic
    """
    logger.info("=" * 60)
    logger.info("Starting Traffic Generator Service")
    logger.info("=" * 60)
    logger.info(f"Configuration:")
    logger.info(f"  - Redis: {REDIS_HOST}:{REDIS_PORT}")
    logger.info(f"  - Queue: {REDIS_QUEUE}")
    logger.info(f"  - Rate: {TRAFFIC_RATE} messages/second")
    logger.info(f"  - Mode: {MODE}")
    logger.info(f"  - Loop: {LOOP}")
    logger.info("=" * 60)
    
    # Step 1: Load data
    data = load_dataset()
    test_data = prepare_test_data(data)
    
    # Step 2: Connect to Redis
    redis_client = create_redis_connection()
    
    # Step 3: Create traffic generator
    traffic_gen = generate_traffic(test_data, mode=MODE)
    
    # Step 4: Start publishing
    publish_traffic(redis_client, traffic_gen)
    
    logger.info("Traffic generator stopped.")


if __name__ == "__main__":
    main()
