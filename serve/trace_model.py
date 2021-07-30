import numpy as np
import torch
import sys

sys.path.append('/home/manhbui/manhbq_workspace/OcrTool/src')
from ManhOCR.tool.config import Cfg
from ManhOCR.tool.translate import build_model, get_bucket, process_image, translate


device = torch.device("cpu")
config = Cfg.load_config_from_file('./src/ManhOCR/config/base.yml')
config['device'] = "cpu"
model, vocab = build_model(config, deploy=True)
model.load_state_dict(torch.load("./src/weights/convert_weight_2.pth.tar", map_location=device), strict=False)

sample = torch.rand(1, 3, 32, 475)
traced_cnn_script_module = torch.jit.trace(model.cnn, sample)
traced_cnn_script_module.save(
    "./serve/traced_scripts/traced_cnn_model.pt")

sample = torch.rand(107, 1, 256)
traced_encoder_script_module = torch.jit.trace(
    model.transformer.encoder, sample)
traced_encoder_script_module.save(
    "./serve/traced_scripts/traced_encoder_model.pt")

sample = (torch.LongTensor([1]), torch.rand(1, 256), torch.rand(107, 1, 512))
traced_decoder_script_module = torch.jit.trace(
    model.transformer.decoder, sample)
traced_decoder_script_module.save(
    "./serve/traced_scripts/traced_decoder_model.pt")