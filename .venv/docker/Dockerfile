# Use a lightweight Python base image
FROM python:3.10-slim

# Set the working directory inside the container and copy the rest of the application code into the container
WORKDIR /inference
COPY . /inference

#Set python path to look for packages
ENV PYTHONPATH=/inference

# Copy the requirements file and install dependencies to leverage Docker cache
RUN pip install --no-cache-dir -r requirements.txt


# Expose the port the application will run on
EXPOSE 8000

# Command to run the application using Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]