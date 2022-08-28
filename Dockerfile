FROM python:3.10-buster

RUN pip3 install torch==1.12.1 torchvision==0.13.0 --extra-index-url https://download.pytorch.org/whl/cpu \ 
        && pip3 install fastapi==0.81.0 python-multipart==0.0.5 uvicorn==0.18.3 opencv-python-headless==4.6.0.66
RUN pip3 install efficientnet_pytorch

ENV APP_DIR=/usr/src/app
ENV PYTHONUNBUFFERED=1

RUN mkdir -p ${APP_DIR}
WORKDIR ${APP_DIR}

COPY ./efficientnet-b1-f1951068.pth /root/.cache/torch/hub/checkpoints/
COPY . ${APP_DIR}
