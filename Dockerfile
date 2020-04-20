FROM pytorch/pytorch:latest
WORKDIR /workdir
COPY . /workdir
RUN pip install -r /workdir/requirements.txt