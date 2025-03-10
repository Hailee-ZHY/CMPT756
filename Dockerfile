
# Use the official lightweight Python image
# https://hub.docker.com/_/python
FROM python:3.10-slim

# Set working directory
WORKDIR /CMPT756

# Copy all local files into the container
COPY . .

# Install dependencies for OpenCV and other system packages
RUN apt update && apt install -y \
    libgl1 libglib2.0-0 wget unzip && \
    rm -rf /var/lib/apt/lists/*

# Make run.sh executable
RUN chmod +x run.sh

# Run run.sh, which installs pip dependencies
CMD ["sh", "-c", "./run.sh"]
