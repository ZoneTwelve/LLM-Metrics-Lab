# Use an official Python base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install dependencies (update pip first)
RUN pip install --upgrade pip

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install -r requirements.txt

# Copy rest of the application
COPY . .

CMD ["sleep", "infinity"]