from collections import defaultdict

import numpy as np
import torch

from .tool.translate import build_model, get_bucket, process_image, translate


class Predictor(object):
    def __init__(
        self, config, device, weight_path="./weights/transformerocr_light_best.pth"
    ):
        model, vocab = build_model(config, deploy=True)
        model.load_state_dict(
            torch.load(weight_path, map_location=device), strict=False
        )

        config["device"] = device
        self.config = config
        self.model = model
        self.vocab = vocab

    def predict(self, img):
        img = process_image(img)
        img = np.expand_dims(img, axis=0)
        img = torch.FloatTensor(img)
        img = img.to(self.config["device"])

        s = translate(img, self.model)[0].tolist()

        s = self.vocab.decode(s)

        return s

    def batch_predict(self, images, batch_size=256):
        """
        param: images : list of ndarray
        """
        cluster_images, indices = self.build_cluster_images(images=images)
        result = list([])

        for cluster, batch_images in cluster_images.items():
            if len(batch_images) <= batch_size:
                batch_images = np.asarray(batch_images)
                batch_images = torch.FloatTensor(batch_images)
                batch_images = batch_images.to(self.config["device"])
                sent = translate(batch_images, self.model).tolist()
                batch_text = self.vocab.batch_decode(sent)
                result.extend(batch_text)
            else:
                for i in range(0, len(batch_images), batch_size):
                    sub_batch_images = torch.FloatTensor(batch_images[i:i + batch_size])
                    sub_batch_images = sub_batch_images.to(self.config['device'])
                    sent = translate(sub_batch_images, self.model).tolist()
                    batch_text = self.vocab.batch_decode(sent)
                    result.extend(batch_text)

        # sort result correspond to indices
        z = zip(result, indices)
        sorted_result = sorted(z, key=lambda x: x[1])
        result, _ = zip(*sorted_result)

        return result

    @staticmethod
    def sort_width(images):
        batch = list(zip(images, range(len(images))))
        sorted_value = sorted(batch, key=lambda x: x[0].shape[1], reverse=False)
        sorted_images, indices = list(zip(*sorted_value))

        return sorted_images, indices

    def build_cluster_images(self, images):
        cluster_images = defaultdict(list)
        sorted_images, indices = self.sort_width(images)

        for img in sorted_images:
            # preprocess
            img, bucket_size = self.process_image(img, img_height=32)
            cluster_images[bucket_size].append(img)

        return cluster_images, indices
