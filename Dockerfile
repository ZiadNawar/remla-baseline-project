FROM python:3.8.4-slim

COPY requirements.txt .
COPY src src
COPY data data

RUN mkdir output && python -m pip install --upgrade pip && pip install -r requirements.txt
RUN python -m nltk.downloader -d /usr/share/nltk_data all
RUN python src/text_preprocessing.py && python src/vectorization.py && python src/model_training.py && python src/evaluation.py
LABEL org.opencontainers.image.source https://github.com/ZiadNawar/remla-baseline-project
LABEL org.opencontainers.image.description Release Engineering for Machine Learning Application (remla) docker image

EXPOSE 8081

ENTRYPOINT ["python"]
CMD ["src/serve_model.py"]