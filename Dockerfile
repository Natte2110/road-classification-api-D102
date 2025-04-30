# Use an official Python runtime as a base image
FROM python:3.11.2-slim

# Set the working directory inside the container
WORKDIR /app

# Copy only the requirements file first (leverages Docker layer caching)
COPY ./app/requirements.txt /app/requirements.txt

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application files
COPY ./app /app

# Expose FastAPI's default port
EXPOSE 8000

# Run FastAPI app with Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
