import json
import time
import random
import logging
from datetime import datetime, timedelta

from kafka import KafkaProducer
from kafka.errors import KafkaError
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NYCTaxiProducer:
    """Producer for NYC Taxi data to Kafka."""

    def __init__(self, bootstrap_servers: str, topic: str, data_path: str):
        self.bootstrap_servers = bootstrap_servers
        self.topic = topic
        self.data_path = data_path

        self.producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode('utf-8'),
            acks='all',
            retries=3,
            max_in_flight_requests_per_connection=1,
        )

    def load_data(self) -> pd.DataFrame:
        """Load NYC Taxi data from parquet file."""
        logger.info(f"Loading data from {self.data_path}")
        df = pd.read_parquet(self.data_path)
        logger.info(f"Loaded {len(df)} records")
        return df

    def inject_anomalies(self, row: dict, anomaly_rate: float = 0.05) -> dict:
        """Inject random anomalies into data stream."""
        if random.random() < anomaly_rate:
            anomaly_types = ['fare_amount', 'trip_distance', 'passenger_count', 'pickup_location']
            anomaly_type = random.choice(anomaly_types)

            if anomaly_type == 'fare_amount':
                row['fare_amount'] = abs(row.get('fare_amount', 10)) * random.uniform(2, 5)
            elif anomaly_type == 'trip_distance':
                row['trip_distance'] = abs(row.get('trip_distance', 2)) * random.uniform(3, 6)
            elif anomaly_type == 'passenger_count':
                row['passenger_count'] = random.choice([0, 10, 15, 99])
            elif anomaly_type == 'pickup_location':
                row['pickup_longitude'] = random.uniform(-180, 180)
                row['pickup_latitude'] = random.uniform(-90, 90)

            row['_is_anomaly_injected'] = True
            row['_anomaly_type'] = anomaly_type
        else:
            row['_is_anomaly_injected'] = False

        return row

    def run(self, rate: float = 0.01, anomaly_rate: float = 0.05):
        """Run the producer loop."""
        df = self.load_data()

        logger.info(f"Starting producer: {len(df)} records at rate {rate}s/record")
        logger.info(f"Anomaly injection rate: {anomaly_rate * 100}%")

        for idx, row in df.iterrows():
            try:
                record = row.to_dict()

                # Add metadata
                record['_producer_timestamp'] = datetime.utcnow().isoformat()
                record['_record_index'] = idx

                # Inject anomalies
                record = self.inject_anomalies(record, anomaly_rate)

                # Send to Kafka
                future = self.producer.send(self.topic, value=record)
                future.get(timeout=10)

                if idx % 1000 == 0:
                    logger.info(f"Sent {idx} records to {self.topic}")

                time.sleep(rate)

            except KafkaError as e:
                logger.error(f"Kafka error at record {idx}: {e}")
                continue
            except Exception as e:
                logger.error(f"Error at record {idx}: {e}")
                continue

        self.producer.flush()
        logger.info(f"Completed: sent {len(df)} records")


def main():
    import os

    bootstrap_servers = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'kafka:9092')
    topic = os.getenv('KAFKA_TOPIC', 'taxi-nyc-raw')
    data_path = os.getenv('DATA_PATH', '/data/yellow_tripdata_2024-01.parquet')
    rate = float(os.getenv('PRODUCE_RATE', '0.001'))
    anomaly_rate = float(os.getenv('ANOMALY_RATE', '0.05'))

    producer = NYCTaxiProducer(
        bootstrap_servers=bootstrap_servers,
        topic=topic,
        data_path=data_path
    )

    producer.run(rate=rate, anomaly_rate=anomaly_rate)


if __name__ == '__main__':
    main()
