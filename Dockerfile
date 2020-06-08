FROM pytorch/pytorch:latest
WORKDIR /workdir
COPY . /workdir
ENV PYTHONPATH /workdir
RUN pip install -r /workdir/requirements.txt