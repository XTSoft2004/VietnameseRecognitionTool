FROM tiangolo/uvicorn-gunicorn-fastapi:python3.8-slim


# working dir
WORKDIR /ocrtool_app

RUN apt-get update

# copy requirements file from local directory
COPY requirements.txt /ocrtool_app/requirements.txt

# install nessecessary library
RUN pip install --upgrade pip
RUN pip install -r /ocrtool_app/requirements.txt

# copy all files in the current directory to working dir
COPY . /ocrtool_app

# add excutable permission
RUN chmod a+x /ocrtool_app/*

# start
ENTRYPOINT ["/ocrtool_app/start.sh"]

# set up working port
EXPOSE 9090






