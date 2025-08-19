# Poker/Dockerfile

FROM python:3.13-slim
WORKDIR /app

# First, install third-party dependencies
COPY requirements.txt .
RUN apt-get update && apt-get install -y \
    fonts-dejavu-core \
    && pip install --no-cache-dir -r requirements.txt \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Next, install your own application as a package
COPY setup.py .
COPY ./app /app/app
RUN pip install -e .

# Copy your models
COPY ./models /app/models

# Expose the port and run the app
EXPOSE 8501
CMD ["streamlit", "run", "app/playgame.py"]