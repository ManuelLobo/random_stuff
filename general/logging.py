import logging
class DataManager:
    def __init__(self, endpoint):
        self.logger = logging.getLogger(
          f"{__name__}.{self.__class__.__name__}")
        self.endpoint = endpoint
    def connect(self):
        self.logger.info(f"Connecting to {self.endpoint}")
...
logging.basicConfig(level=logging.DEBUG,
  format="[%(asctime)s] [%(processName)s:%(threadName)s] "
         "[test-project/%(name)s.%(funcName)s:%(lineno)d] "
         "[%(levelname)s] %(message)s")
