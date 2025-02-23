# Use an official Python runtime as a parent image.
FROM python:3.9-slim

# Set environment variables to improve Python output and buffering.
ENV PYTHONUNBUFFERED=1

# Set the working directory in the container.
WORKDIR /app

# Copy only requirements.txt first to leverage Docker cache.
COPY requirements.txt /app/

# Install the dependencies.
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy the rest of the application code.
COPY . /app

# Expose the port on which the app will run.
EXPOSE 8000

# Run the FastAPI app with Uvicorn.
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
