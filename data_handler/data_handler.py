from data_handler.data_producer import DataProducer
from utils import *

class DataHandler:
    def __init__(self, config):
        self.data_producer = DataProducer(config)
        self.data_producer.start()

    def next_batch(self, status):
        return self.data_producer.consume(status)

    def write_image(self):
        pass

    def get_valtest_size(self):
        return self.data_producer.get_valtest_size()

    def get_bags_per_batch(self):
        return self.data_producer.get_bags_per_batch()