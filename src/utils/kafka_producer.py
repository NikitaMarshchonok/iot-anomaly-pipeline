'''
Tiny Kafka producer for stack test.
Sends 1 JSON message per second to topic ‘sensor-data’.
'''


import json
import random
import time
from datetime import datetime

from kafka import KafkaProducer

# ##### 1. Инициализация продюсера ########################################
# bootstrap_servers → адрес брокера (у нас контейнер kafka:9092,
# но с хоста обращаемся к localhost:9092, потому что порт проброшен наружу)

producer = KafkaProducer(
    bootstrap_servers="localhost:9092",
    value_serializer=lambda v: json.dumps(v).encode("utf-8"),
)

# ##### 2. Бесконечный цикл отправки #######################################
while True:
    payload = {
        "ts": datetime.utcnow().isoformat(timespec="seconds"),
        "value": round(random.random(), 4),
    }
    future = producer.send("sensor-data", payload)
    # .get() заставляет дождаться подтверждения, чтобы увидеть ошибки сразу
    result_metadata = future.get(timeout=10)

    print(f"sent → partition={result_metadata.partition} offset={result_metadata.offset}  {payload}")
    time.sleep(1)