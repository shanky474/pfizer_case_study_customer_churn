# Use a lightweight Python base image
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /inference

# Copy the requirements file and install dependencies to leverage Docker cache
COPY .venv/Scripts/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY .venv/Scripts/artifacts .
COPY .venv/Lib/site-packages/utils .
COPY .venv/Scripts/src/inference/main.py .

# Expose the port the application will run on
EXPOSE 8000

# Command to run the application using Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]