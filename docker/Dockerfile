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
#Workaround porque instala tf-2 no-gpu https://github.com/tensorflow/text/issues/195
#RUN pip install tensorflow-text
#RUN pip uninstall -y tensorflow

#RUN jupyter notebook --generate-config
#RUN echo "c.NotebookApp.password='sha1:3e8f1e7d4570:9e684db783a0ff0f73590f960ab2a69b9f6e9fe7'" >> /root/jupyter_notebook_config.py

