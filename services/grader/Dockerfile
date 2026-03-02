FROM stereolabs/zed:5.2-py-runtime-jetson-jp5.1.2

WORKDIR /opt/grader

# Install PyTorch for Jetson (JetPack 5.x / L4T R35)
RUN pip3 install --no-cache-dir \
    https://developer.download.nvidia.com/compute/redist/jp/v512/pytorch/torch-2.1.0a0+41361538.nv23.06-cp38-cp38-linux_aarch64.whl \
    || pip3 install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install Python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ app/

ENV PYTHONUNBUFFERED=1

CMD ["python3", "-m", "app.worker"]
