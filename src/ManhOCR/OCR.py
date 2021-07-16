from torch import nn

from .model.cnn import CNN
from .model.seq2seq import Seq2Seq


class OCR(nn.Module):
    def __init__(self, vocab_size, config, deploy=False):
        super(OCR, self).__init__()
        self.config = config
        self.deploy = deploy
        self.cnn = CNN(self.config["model_type"], deploy=self.deploy)

        self.transformer = Seq2Seq(
            vocab_size,
            encoder_hidden=config["seq_parameters"]["encoder_hidden"],
            decoder_hidden=config["seq_parameters"]["decoder_hidden"],
            img_channel=config["seq_parameters"]["img_channel"],
            decoder_embedded=config["seq_parameters"]["decoder_embedded"],
            dropout=config["seq_parameters"]["dropout"],
        )

    def forward(self, img, tgt_input, tgt_key_padding_mask=None):
        """
        Shape:
            - img: (N, C, H, W)
            - tgt_input: (T, N)
            - tgt_key_padding_mask: (N, T)
            - output: b t v
        """
        src = self.cnn(img)
        outputs = self.transformer(src, tgt_input)

        return outputs
