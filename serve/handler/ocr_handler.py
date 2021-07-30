from numpy.core.numeric import indices
from numpy.lib.type_check import imag
from ts.torch_handler.base_handler import BaseHandler
import os
import torch
import logging
import numpy as np
import cv2
import zipfile
from collections import defaultdict
logger = logging.getLogger(__name__)
from config import Cfg
from vocab import Vocab
import base64
from PIL import Image
import io
import json

class OCRHandler(BaseHandler):
    def __init__(self):
        super().__init__()
        self._context = None
        self.initialized = False
        self.batch_size = 256

    def inference(self, data, *args, **kwargs):
        return super().inference(data, *args, **kwargs)

    def initialize(self, context):
        """Initialize function loads the model.pt file and initialized the model object.
           First try to load torchscript else load eager mode state_dict based model.

        Args:
            context (context): It is a JSON Object containing information
            pertaining to the model artifacts parameters.

        Raises:
            RuntimeError: Raises the Runtime error when the model.py is missing

        """
        properties = context.system_properties
        self.map_location = (
            "cuda"
            if torch.cuda.is_available() and properties.get("gpu_id") is not None
            else "cpu"
        )
        self.device = torch.device("cpu")
        self.manifest = context.manifest


        config = Cfg.load_config_from_file('base.yml')
        self.vocab = Vocab(config["vocab"])

        model_dir = properties.get("model_dir")
        cnn_weight_path = os.path.join(model_dir, 'cnn_model.pt')
        decoder_weight_path = os.path.join(model_dir, 'decoder_model.pt')
        encoder_weight_path = os.path.join(model_dir, 'encoder_model.pt')

        try:
            self.cnn = torch.jit.load(cnn_weight_path, map_location=self.device)
            self.encoder = torch.jit.load(encoder_weight_path, map_location=self.device)
            self.decoder = torch.jit.load(decoder_weight_path, map_location=self.device)
            self.initialized = True
        except:
            self.initialized = False


    def preprocess(self, requests):
        if len(requests) > 1:
            images = [self.preprocess_one_image(req) for req in requests]
            cluster_images, indices = self.build_cluster_images(images)
            return cluster_images, indices
        else:
            image = self.preprocess_one_image(requests[0])
            return image, None

    def preprocess_one_image(self, req):
        """
        Process one single image
        """
        print('Request: ', req)
        b64_code = req.get('data')
        if b64_code is None:
            b64_code = req.get('body')

        # create a stream from the encoded image
       # Restore OpenCV image from base64 encoding
        str_decode = base64.b64decode(b64_code)
        nparr = np.fromstring(str_decode, np.uint8)
        # img_restore = cv2.imdecode(nparr, cv2.CV_LOAD_IMAGE_COLOR) for python 2
        img_restore = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        image = np.array(img_restore)

        return image

    def inference(self, data, *args, **kwargs):
        images, indices = data

        if indices:
            result = list([])
            for _, batch_images in images.items():
                if len(batch_images) <= self.batch_size:
                    batch_images = np.asarray(batch_images)
                    batch_images = torch.FloatTensor(batch_images)
                    batch_images = batch_images.to(self.device)

                    sent = self.translate(batch_images).tolist()
                    batch_text = self.vocab.batch_decode(sent)
                    result.extend(batch_text)
                else:
                    for i in range(0, len(batch_images), self.batch_size):
                        sub_batch_images = torch.FloatTensor(batch_images[i:i + self.batch_size])
                        sub_batch_images = sub_batch_images.to(self.config['device'])
                        sent = self.translate(sub_batch_images).tolist()
                        batch_text = self.vocab.batch_decode(sent)
                        result.extend(batch_text)

            # sort result correspond to indices
            z = zip(result, indices)
            sorted_result = sorted(z, key=lambda x: x[1])
            result, _ = zip(*sorted_result)
        else:
            image, _ = self.process_image(images)
            image = np.expand_dims(image, axis=0)
            image = torch.FloatTensor(image).to(device=self.device)
            result = self.translate(image)[0].tolist()
            result = self.vocab.decode(result)

        return result

    def postprocess(self, data):
        return [data]

    def translate(self, tensor, max_seq_length=128, sos_token=1, eos_token=2):
        """data: BxCXHxW"""
        device = tensor.device

        with torch.no_grad():
            src = self.cnn(tensor)
            encoder_outputs, hidden = self.encoder(src)

            translated_sentence = [[sos_token] * len(tensor)]
            max_length = 0

            while max_length <= max_seq_length and not all(
                np.any(np.asarray(translated_sentence).T == eos_token, axis=1)
            ):
                tgt_inp = torch.LongTensor(translated_sentence).to(device)
                tgt = tgt_inp[-1]
                output, hidden, _ = self.decoder(tgt, hidden, encoder_outputs)
                output = output.unsqueeze(1)

                output = output.to("cpu")

                _, indices = torch.topk(output, 5)

                indices = indices[:, -1, 0]
                indices = indices.tolist()

                translated_sentence.append(indices)
                max_length += 1
                del output

            translated_sentence = np.asarray(translated_sentence).T

        return translated_sentence


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


    def process_image(self, img, img_height=32):
        # convert to numpy array
        img = np.asarray(img)
        w = img.shape[1]
        
        # get bucket size
        bucket_size = self.get_bucket(w)

        # padding image:
        padding_right = bucket_size - w
        
        if padding_right < 0:
            img = cv2.resize(img, (bucket_size, img_height), cv2.INTER_AREA)
        else:
            img = cv2.resize(img, (w, img_height), cv2.INTER_AREA)
            img = cv2.copyMakeBorder(img, top=0, left=0, right=padding_right, bottom=0, borderType=cv2.BORDER_CONSTANT, value=0)
    
        img = img.transpose(2, 0, 1)
        img = img / 255.0

        return img, bucket_size


    def get_bucket(self, w):
        if w > 900:
            bucket_size = 900
        elif w > 800:
            bucket_size = 800
        elif w < 80:
            bucket_size = 70
        elif w < 120:
            bucket_size = (w // 20) * 20
        else:
            bucket_size = (w // 30) * 30
            
        return bucket_size

