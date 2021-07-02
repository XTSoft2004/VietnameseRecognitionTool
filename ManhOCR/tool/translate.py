import cv2
import numpy as np
import torch

from ..model.vocab import Vocab
from ..OCR import OCR


def translate(img, model, max_seq_length=256, sos_token=1, eos_token=2):
    """data: BxCXHxW"""
    model.eval()
    device = img.device

    with torch.no_grad():
        src = model.cnn(img)
        memory = model.transformer.forward_encoder(src)

        translated_sentence = [[sos_token] * len(img)]
        max_length = 0

        while max_length <= max_seq_length and not all(
            np.any(np.asarray(translated_sentence).T == eos_token, axis=1)
        ):
            tgt_inp = torch.LongTensor(translated_sentence).to(device)
            output, memory = model.transformer.forward_decoder(tgt_inp, memory)
            output = output.to("cpu")

            values, indices = torch.topk(output, 1)

            indices = indices[:, -1, 0]
            indices = indices.tolist()

            translated_sentence.append(indices)
            max_length += 1

            del output

        translated_sentence = np.asarray(translated_sentence).T

    return translated_sentence


def build_model(config, deploy=False):
    vocab = Vocab(config["vocab"])
    device = config["device"]

    model = OCR(len(vocab), config, deploy=deploy)
    model = model.to(device)

    return model, vocab


def get_bucket(w):
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


def process_image(image, img_height=32):
    # convert to numpy array
    img = np.asarray(image)
    w = img.shape[1]
    
    # get bucket size
    bucket_size = get_bucket(w)

    # padding image:
    padding_right = bucket_size - w
    
    if padding_right < 0:
        img = cv2.resize(img, (bucket_size, img_height), cv2.INTER_AREA)
    else:
        img = cv2.resize(img, (w, img_height), cv2.INTER_AREA)
        img = cv2.copyMakeBorder(img, top=0, left=0, right=padding_right, bottom=0, borderType=cv2.BORDER_CONSTANT, value=0)
   
    img = img.transpose(2, 0, 1)
    img = img / 255.0

    return img