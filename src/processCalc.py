from domain.config import Config
from src.calculateInstantly import calculate

def calculate_task(segments, config:Config):
    return calculate(segments, config)
