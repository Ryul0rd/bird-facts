# 1. Base image
FROM python:3.8.5-slim-buster

# 2. Get requirements
COPY requirements.txt .

# 3. Install dependencies
RUN pip install -r requirements.txt

# 4. Copy files
COPY . /src

# 5. Start app
ENTRYPOINT ["python", "/src/bird-facts.py"]