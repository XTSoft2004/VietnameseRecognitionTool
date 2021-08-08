## 1. Archive model
```
torch-model-archiver --model-name ocr_model --serialized-file ./serve/traced_scripts/model_scripts.zip --version 1.0  --export-path /home/manhbui/manhbq_workspace/VietnameseRecognitionTool/serve/model_store --handler ./serve/handler/ocr_handler.py --extra-files ./src/ManhOCR/tool/config.py,./src/ManhOCR/model/vocab.py,./src/ManhOCR/config/base.yml,./serve/traced_scripts/model_scripts/cnn_model.pt,./serve/traced_scripts/model_scripts/decoder_model.pt,./serve/traced_scripts/model_scripts/encoder_model.pt
```

## 2. Serve model
```
torchserve --start --ncs --model-store ./serve/model_store --models ./serve/model_store/ocr_model.mar --foreground --ts-config ./serve/config.properties
```
## 3. Run demo app
```
streamlit run app.py
```


## 4. Inference - Using Rest Api
```
curl http://127.0.0.1:8085/predictions/ocr_model -T ./sample/0.png
```


## 5. Some command line related to torchserve 
Link: https://pytorch.org/serve/management_api.html
- Register new model
```
curl -X POST "http://localhost:8081/models?url=model_name.mar"
```
- Show the list of registered models:
```
curl "http://localhost:8081/models"
```
- Set minium the number of worker
```
curl -v -X PUT "http://localhost:8081/models/model_name?min_worker=2"
curl "http://localhost:8081/models/model_name"
```
- Unregister model
```
curl -X DELETE http://localhost:8081/models/model_name/
```
- Stop
```
torchserve --stop
```
- Verify that TorchServe is up and running
```
curl localhost:8080/ping
```

## 6. Using torchserve-dashboard
Run: 
```
torchserve-dashboard --server.port 8505 -- --config_path ./config.properties
```

## 7. Run using docker
