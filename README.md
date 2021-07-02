## 1. Archive model
```
torch-model-archiver --model-name ocr_model --serialized-file ./traced_scripts/model_scripts.zip --version 1.0  --export-path model_store --handler ./handler/ocr_handler.py --extra-files ./ManhOCR/tool/config.py,./ManhOCR/model/vocab.py,./ManhOCR/config/base.yml,./traced_scripts/model_scripts/cnn_model.pt,./traced_scripts/model_scripts/decoder_model.pt,./traced_scripts/model_scripts/encoder_model.pt
```

## 2. Serve model
```
torchserve --ncs --model-store model_store --models all
```
## 3. Inference 
### 3.1. Using Rest Api
```
curl http://127.0.0.1:8080/predictions/ocr_model -T ./sample/0.png
```


## 4. Some command line related to torchserve
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