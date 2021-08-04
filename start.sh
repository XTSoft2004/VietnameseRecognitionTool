#!/bin/bash
torchserve --start --ncs --model-store ./serve/model_store --models ./serve/model_store/ocr_model.mar --foreground --ts-config ./serve/config.properties
# streamlit run app.py
exec "$@"
