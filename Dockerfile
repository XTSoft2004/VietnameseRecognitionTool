
FROM pytorch/torchserve:0.1-cpu

# copy all files in the current directory to working dir
COPY . .

# install nessecessary library
RUN pip install --upgrade pip
RUN pip install -r /home/model-server/requirements.txt
RUN pip install torch-model-archiver -q

USER root
# add excutable permission
EXPOSE 8080 8081


