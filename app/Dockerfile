FROM python:3.9-slim-buster

# Set the working directory inside the container
WORKDIR /app

# Install necessary packages and configure timezone
RUN apt-get update && apt-get install -y procps && pip install -U pip \
    && rm /etc/localtime \
    && ln -s /usr/share/zoneinfo/America/Mexico_City /etc/localtime  \
    && apt-get install libgomp1

# Copy the requirements file into the container and install dependencies
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY main.py /app

# Expose port 8000
EXPOSE 8000

# Set the command to run FastAPI using uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]