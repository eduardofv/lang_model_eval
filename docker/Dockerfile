FROM tensorflow/tensorflow:latest-gpu-py3-jupyter
WORKDIR /root

EXPOSE 80
#JUPYTER
EXPOSE 8888
#TENSORBOARD 
EXPOSE 6006

ENV SHELL /bin/bash
ENV PATH="/root/.local/bin:${PATH}"
ENV TFHUB_CACHE_DIR="/root/cache/tfhub"
ENV CUDA_CACHE_PATH="/root/cache/cuda"

#ENV TZ=America/Mexico_City
#RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN ln -sf /usr/share/zoneinfo/America/Mexico_City /etc/localtime

RUN apt install -y zip
RUN apt install -y git
RUN pip install pandas
RUN pip install matplotlib
RUN pip install sklearn
RUN pip install tqdm
RUN pip install tensorflow-hub
RUN pip install keras
RUN pip install tensorflow-text==2.1.0-rc0
RUN pip install pytest
RUN pip install sentencepiece
RUN pip install bert-for-tf2
RUN pip install transformers
RUN pip install seaborn

ENV PYTORCH_PRETRAINED_BERT_CACHE="/root/cache/hug"

RUN apt install -y vim 
RUN pip install torch==1.4.0+cpu torchvision==0.5.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
