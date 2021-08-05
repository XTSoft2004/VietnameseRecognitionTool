
FROM pytorch/torchserve:0.1-cpu

USER root

COPY requirements.txt .
RUN apt-get update && apt-get install -y libglib2.0-0 libsm6 libxrender1 libxext6 -y

# RUN pip install --upgrade pip
RUN pip install -r /home/model-server/requirements.txt
COPY . .

USER model-server
EXPOSE 9090

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
ENV PYTHONIOENCODING=utf-8
