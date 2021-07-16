import yaml

from .predictor import Predictor
from .tool.config import Cfg


class TextRecognition(object):
    def __init__(self, device, weight_path, config_path, model_type):
        self.config = Cfg.load_config_from_file(config_path)
        self.config["device"] = device
        self.config["model_type"] = model_type
        self.predictor = Predictor(self.config, device=device, weight_path=weight_path)

    @staticmethod
    def read_from_config(file_yml):
        with open(file_yml, encoding="utf-8") as f:
            config = yaml.safe_load(f)

        return config

    def predict(self, image):
        result = self.predictor.predict(image)

        return result

    def predict_on_batch(self, batch_images, batch_size=256):
        return self.predictor.batch_predict(batch_images, batch_size)
