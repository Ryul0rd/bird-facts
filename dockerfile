# 1. Base image
FROM python:3.8.5-slim-buster

# 2. Copy files
COPY . /src

# 3. Install dependencies
RUN pip install -r /src/requirements.txt