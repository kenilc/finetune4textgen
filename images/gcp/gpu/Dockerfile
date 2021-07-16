FROM gcr.io/cloud-ml-public/training/pytorch-gpu.1-7

ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

WORKDIR /root

COPY ./trainer/ ./trainer/
COPY ./requirements.txt ./requirements.txt

RUN pip install --upgrade pip
RUN pip install --upgrade -r ./requirements.txt

ENTRYPOINT ["python", "-m", "trainer.task"]
