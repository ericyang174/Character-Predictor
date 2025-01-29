# TODO: set up docker properly
FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime
RUN mkdir /job
WORKDIR /job
VOLUME ["/job/data", "/job/src", "/job/work", "/job/output"]

# RUN pip install python==3.9
# RUN pip install numpy==1.21
# RUN pip install pandas==1.3.3
# RUN pip install pytorch==1.9.0
# RUN pip install tqdm==4.62.3
# RUN pip install matplotlib==3.4.3
# RUN pip install pyyaml