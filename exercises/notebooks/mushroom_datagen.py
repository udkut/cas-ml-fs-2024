import argparse
import sys
import time
from random import randrange


import pandas as pd
from kafka import KafkaProducer
from kafka.errors import KafkaError
from loguru import logger
from sklearn.neighbors import KernelDensity

producer = KafkaProducer(
    bootstrap_servers = "message-broker:9092",
)

def on_success(metadata):
    logger.debug(f"Message produced to topic '{metadata.topic}' at offset {metadata.offset}")


def on_error(e):
    logger.error(f"Error sending message: {e}")


def setup_data():
    # fetch Training Dataset so we have a reference
    df = pd.read_parquet('s3://traindata/train_raw.parquet',
                         storage_options={"anon": False}).drop("class", axis="columns")

    # setup column names
    categoricals = ['cap-shape', 'gill-attachment', 'gill-color', 'stem-color']
    numericals = [c for c in df.columns if c not in categoricals]

    # fit an estimator the the numerical columns
    kde = KernelDensity()
    kde.fit(df[numericals])

    return kde, numericals, categoricals, df


def generate_event(kde, numericals, categoricals, df,drift):
    # take one row so we have something to fill in our generated values
    new_row = pd.DataFrame(data=df.head(1))

    # generate one row
    new_row[numericals] = kde.sample(1)
    for col in numericals:
        # we are being lazy, when the kde yields a negative, just replace with mean
        if new_row[col][0] < 0:
            new_row[col] = df[col].mean()
    for col in categoricals:
        # for the categoricals, use a bounded random value
        new_row[col] = randrange(df[col].min(), df[col].max()+1)
    for col in new_row.columns:
        # make sure the datatypes match the reference
        new_row[col] = new_row[col].astype(df[col].dtype)

    new_row['season'] = new_row['season']*drift

    return new_row.iloc[0].to_json()


def push_to_kafka(event, topic):
    # send to redpanda
    future = producer.send(topic=topic, key=b'key', value=event.encode('utf-8'))  # we skip message serializer and schema checks here
    future.add_callback(on_success)
    future.add_errback(on_error)
    producer.flush()


def run(topic, sleep_interval, burst_size, randomness, drift):
    kde, numericals, categoricals, df = setup_data()
    while True:
        for _ in range(burst_size):
            push_to_kafka(event=generate_event(kde, numericals, categoricals, df, drift), topic=topic)

        try:
            sleep_milliseconds = sleep_interval + randrange(0, randomness*1000)
        except ValueError:
            # if randomness*1000 < 1
            sleep_milliseconds = sleep_interval
        time.sleep(sleep_milliseconds / 1000)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Simulates inference requests by sending mushroom dataset features to kafka."
    )
    parser.add_argument(
        "-t",
        "--topic",
        type=str,
        help="Kafka topic to send to",
        default="mushroom_inference_request",
    )
    parser.add_argument(
        "-s",
        "--sleep_interval",
        type=int,
        help="Number of milliseconds to sleep between bursts",
        default=1000,
    )
    parser.add_argument(
        "-b",
        "--burst_size",
        type=int,
        help="Number of messages to send together",
        default=1,
    )
    parser.add_argument(
        "-r",
        "--randomness",
        type=int,
        help="Maximum number of seconds to wait between bursts",
        default=1,
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action='store_true',
        help="Log content of each message sent sto stderr",
    )
    parser.add_argument(
        "-d",
        "--drift_factor",
        type=int,
        help="Start factor for simulating drift",
        default=1,
    )
    args = parser.parse_args()

    if not args.verbose:
        # increase loglevel from the default DEBUG to INFO to avoid logging every message
        logger.remove(0)
        logger.add(sys.stderr, level="INFO")

    logger.info("Start sending messages to kafka.")
    run(
        topic = args.topic,
        sleep_interval = args.sleep_interval,
        burst_size = args.burst_size,
        randomness = args.randomness,
        drift = args.drift_factor
    )