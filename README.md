## 1. Demo
<p align="center">
    <img src="https://images.viblo.asia/3b3e0702-7e9c-45f5-9e6d-871d0b3364f0.png" >
Image Text Recognition Tool
</p>

## 2. Deploy by docker

### 2.1. Build streamlit image separately

```
docker build -t streamlit -f ./DockerfileStreamlit .
```

### 2.2 Build and start server 
```
docker-compose up
```
You can view my Streamlit app in your browser:

- Network URL: http://localhost:8501/
- External URL: http://42.114.103.105:8501