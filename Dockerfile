FROM zironycho/pytorch:1120-cpu-py38

ENV GRADIO_SERVER_PORT 7860

COPY ./model.script.pt.tar.gz .
COPY src/demo.py .

COPY requirements.txt .

RUN pip install -r requirements.txt \
    && rm -rf /root/.cache/pip


EXPOSE 7860

CMD ["python", "demo.py"]