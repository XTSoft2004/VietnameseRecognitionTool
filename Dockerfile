FROM pytorch/torchserve:0.1-cpu

# working dir
WORKDIR /ocrtool_app_ver1

# copy requirements file from local directory
COPY requirements.txt /ocrtool_app_ver1/requirements.txt

# install nessecessary library
RUN pip install --upgrade pip
RUN pip install -r /ocrtool_app_ver1/requirements.txt

# copy all files in the current directory to working dir
COPY . /ocrtool_app_ver1

# add excutable permission
# RUN chmod a+x /ocrtool_app_ver1/*

# start
ENTRYPOINT ["/ocrtool_app_ver1/start.sh"]

# set up working port
EXPOSE 9090






