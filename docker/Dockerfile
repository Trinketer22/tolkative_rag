FROM python:3.12-slim AS builder

ENV VENV=/opt/venv
RUN python -m venv $VENV
ENV PATH="${VENV}/bin:$PATH"
ENV PIP_EXTRA_INDEX_URL="https://download.pytorch.org/whl/cpu"

COPY requirements.txt .
RUN pip install -r requirements.txt
#RUN pip freeze

FROM python:3.12-slim AS app
ARG USER_ID=1003
ARG GROUP_ID=1003

RUN groupadd -g ${GROUP_ID} appuser && \
    useradd -m -u ${USER_ID} -g appuser appuser

WORKDIR /app
RUN mkdir logs
RUN chown -R appuser:appuser .

ENV VENV=/opt/venv
ENV PATH="${VENV}/bin:$PATH"
ENV PYTHONPATH=/app

COPY --from=builder $VENV $VENV

COPY --chown=appuser:appuser requirements.txt requirements.txt
COPY --chown=appuser:appuser pyproject.toml pyproject.toml
COPY --chown=appuser:appuser *.py .
COPY --chown=appuser:appuser core core
COPY --chown=appuser:appuser scripts scripts
COPY --chown=appuser:appuser services services
COPY --chown=appuser:appuser models models
COPY --chown=appuser:appuser query query
COPY --chown=appuser:appuser api api
COPY --chown=appuser:appuser utils utils
COPY --chown=appuser:appuser rag-data rag-data

# Installs the rag-setup
RUN pip install --no-deps .

USER appuser

RUN rag-setup rag-data/docs.jsonl
CMD ["python", "main.py"]
