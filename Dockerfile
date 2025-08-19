# Poker/Dockerfile

# 1. Start from a lean, official Python base image.
FROM python:3.9-slim

# 2. Set the working directory inside the container to /app.
WORKDIR /app

# 3. Copy and install dependencies. This is done first to leverage
#    Docker's layer caching, speeding up future builds.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy your entire application source code into the container.
#    This includes playgame.py, visuals/, analyzers/, etc.
COPY ./app .

# 5. Copy your pre-trained models into a 'models' directory
#    inside the container, where your code expects to find them.
COPY ./models /app/models

# 6. Set the default command to run when the container starts.
#    This will launch your game.
CMD ["python", "playgame.py"]
