FROM python:3.11

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements_docker.txt

COPY . .

# Verify the DNN face detector files are present at build time
# so the image fails fast rather than at runtime
RUN test -f app/deploy.prototxt \
    || (echo "ERROR: app/deploy.prototxt not found" && exit 1)
RUN test -f app/res10_300x300_ssd_iter_140000_fp16.caffemodel \
    || (echo "ERROR: app/res10_300x300_ssd_iter_140000_fp16.caffemodel not found" && exit 1)

EXPOSE 8000

CMD ["uvicorn", "app.app:app", "--host", "0.0.0.0", "--port", "8000"]