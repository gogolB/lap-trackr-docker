# Jetson AGX Orin Setup

This guide covers setting up Lap-Trackr on a fresh Jetson AGX Orin from scratch.

## Hardware Requirements

- **Jetson AGX Orin** (64GB recommended)
- **JetPack 5.1.2** (L4T R35.4.1) or later
- **Two ZED X stereo cameras** (GMSL2 interface)
- **NVMe SSD** (500 GB+ recommended) -- the built-in eMMC is only ~57 GB and insufficient for Docker images + session data
- **Ethernet or Wi-Fi** for initial setup

## Phase 0: Initial System Setup

### 1. Flash JetPack

Follow NVIDIA's [SDK Manager instructions](https://developer.nvidia.com/sdk-manager) to flash JetPack 5.1.2+ onto the Jetson. After flashing, complete the initial Ubuntu setup wizard.

### 2. Format and Mount the NVMe SSD

The NVMe drive needs to be formatted and mounted before Docker or data can be stored on it.

```bash
# Identify the NVMe device (usually /dev/nvme0n1)
lsblk

# Create a single partition (CAUTION: destroys all data on the drive)
sudo parted /dev/nvme0n1 --script mklabel gpt
sudo parted /dev/nvme0n1 --script mkpart primary ext4 0% 100%

# Format as ext4
sudo mkfs.ext4 /dev/nvme0n1p1

# Create mount point and mount
sudo mkdir -p /data
sudo mount /dev/nvme0n1p1 /data

# Make it permanent (add to fstab)
echo "$(blkid -s UUID -o value /dev/nvme0n1p1) /data ext4 defaults,noatime 0 2" | sudo tee -a /etc/fstab

# Verify
df -h /data
```

### 3. Install Docker

```bash
# Install Docker Engine
sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
```

### 4. Move Docker Data Root to NVMe

The default Docker data root (`/var/lib/docker`) lives on the small eMMC. Move it to the NVMe:

```bash
sudo systemctl stop docker

# Create Docker config
sudo mkdir -p /etc/docker
sudo tee /etc/docker/daemon.json <<EOF
{
  "data-root": "/data/docker",
  "runtimes": {
    "nvidia": {
      "path": "nvidia-container-runtime",
      "runtimeArgs": []
    }
  },
  "default-runtime": "nvidia"
}
EOF

sudo systemctl start docker
```

### 5. Install NVIDIA Container Toolkit

```bash
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### 6. Install ZED SDK

Install the ZED SDK on the **host** system. The Docker containers mount `/usr/local/zed` from the host.

Download the ZED SDK for JetPack 5 from [Stereolabs downloads](https://www.stereolabs.com/developers/release) and run the installer:

```bash
chmod +x ZED_SDK_Tegra_JP51_v4.1.4.run
./ZED_SDK_Tegra_JP51_v4.1.4.run
```

### 7. Add Your User to the Docker Group

```bash
sudo usermod -aG docker $USER
# Log out and back in for group change to take effect
```

### 8. Create Data Directories

```bash
sudo mkdir -p /data/postgres /data/redis /data/models /data/calibration /data/users
sudo chown -R $USER:$USER /data/users /data/models /data/calibration
```

## Phase 1: Application Setup

### 1. Clone the Repository

```bash
cd /home/$USER
git clone <repo-url> lap-trackr
cd lap-trackr
```

### 2. Configure Environment

```bash
cp .env.example .env
```

Edit `.env` with your settings:

```bash
# Generate a secure JWT secret
python3 -c "import secrets; print(secrets.token_urlsafe(32))"
```

Key values to set in `.env`:

```env
# Database (change the password!)
POSTGRES_PASSWORD=<strong-password>
DATABASE_URL=postgresql+asyncpg://laptrackr:<strong-password>@db:5432/laptrackr

# Auth (paste the generated secret)
JWT_SECRET=<paste-generated-secret>

# Camera serials (find via ZED Explorer or plug in cameras and check dmesg)
ZED_SERIAL_ON_AXIS=<serial-number>
ZED_SERIAL_OFF_AXIS=<serial-number>
```

### Finding Camera Serial Numbers

With cameras connected:

```bash
# Option 1: ZED Explorer (GUI)
/usr/local/zed/tools/ZED_Explorer

# Option 2: ZED Diagnostic (CLI)
/usr/local/zed/tools/ZED_Diagnostic -a

# Option 3: Start the system without serials, then check the camera list endpoint
curl http://localhost:8001/cameras
```

### 3. Build and Start

```bash
docker compose build
docker compose up -d
```

First build takes 15-30 minutes (downloads ZED SDK base images, compiles PyTorch wheels).

### 4. Verify

```bash
# Check all services are healthy
docker compose ps

# Check system health via the API
curl http://localhost/api/health/system | python3 -m json.tool

# Open the web UI
# Navigate to http://<jetson-ip> in a browser
```

### 5. Create a User Account

Open `http://<jetson-ip>` in a browser and click "Register" to create your first account.

## Phase 2: Camera Calibration

See [Calibration Guide](calibration.md) for the full workflow. In short:

1. Print a ChArUco board at actual size
2. Go to **Live View** > **Calibration** panel
3. Capture 5+ frames with the board at varied angles
4. Click **Compute** for each camera
5. Click **Compute Stereo** for the inter-camera transform
6. Verify reprojection error is < 1.0 px

## Maintenance

### Updating

```bash
cd ~/lap-trackr
git pull
docker compose build
docker compose up -d
```

### Viewing Logs

```bash
# All services
docker compose logs -f

# Specific service
docker compose logs -f api
docker compose logs -f grader
docker compose logs -f exporter
```

### Database Backup

```bash
docker compose exec db pg_dump -U laptrackr laptrackr > backup_$(date +%Y%m%d).sql
```

### Database Restore

```bash
cat backup_20260308.sql | docker compose exec -T db psql -U laptrackr laptrackr
```

### Resetting Everything

```bash
# Stop and remove containers + volumes (ALL DATA LOST)
docker compose down -v

# Also clear session data
sudo rm -rf /data/users/* /data/models/* /data/calibration/*
```

### Checking Disk Usage

```bash
# Overall NVMe usage
df -h /data

# Docker images
docker system df

# Session data
du -sh /data/users/*

# Clean unused Docker images
docker system prune -f
```

## Performance Notes

- The Jetson AGX Orin 64GB can comfortably run all 8 services simultaneously
- SVO2 export uses hardware NVENC encoding (much faster than software)
- Depth estimation uses ZED SDK neural mode (GPU-accelerated)
- ML backends (YOLO, CoTracker, SAM2, TAPIR) share GPU memory; only one model is loaded at a time
- Frame sampling (every 5th frame by default) reduces processing time and memory usage
