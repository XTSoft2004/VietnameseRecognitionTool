## 1. Demo
<p align="center">
    <img src="https://images.viblo.asia/3b3e0702-7e9c-45f5-9e6d-871d0b3364f0.png" >
Image Text Recognition Tool
</p>

## 2. Deploy by docker
### 2.1. Check your ip address in ubuntu 
```
ip addr show
```
ipv4 address begin with 192.168.x.x

Then, replace the value of ip_addr in file [app.py](https://github.com/buiquangmanhhp1999/VietnameseRecognitionTool/blob/develop/app.py) by your ip address
```python
ip_addr = "192.168.1.187"
```
### 2.2. Build streamlit image separately

```
docker build -t streamlit -f ./DockerfileStreamlit .
```

### 2.3 Build and start server 
```
docker-compose up
```
You can view my Streamlit app in your browser:

- Network URL: http://localhost:8501/
- External URL: http://42.114.103.105:8501