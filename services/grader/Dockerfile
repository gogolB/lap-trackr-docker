FROM stereolabs/zed:4.1-py-runtime-jetson-jp5.1.2

WORKDIR /opt/grader

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ app/

ENV PYTHONUNBUFFERED=1

CMD ["python", "-m", "app.worker"]
