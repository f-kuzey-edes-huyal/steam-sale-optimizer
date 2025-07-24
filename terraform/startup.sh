#!/bin/bash
# Install Docker & Docker Compose
apt-get update -y
apt-get install -y docker.io docker-compose.azure git curl

# Allow docker without sudo
usermod -aG docker azureuser

# Clone your repo
git clone https://github.com/f-kuzey-edes-huyal/steam-sale-optimizer.git /home/azureuser/steam-sale-optimizer

# Fix permissions
chown -R azureuser:azureuser /home/azureuser/steam-sale-optimizer

# Run containers
cd /home/azureuser/steam-sale-optimizer
docker-compose.azure up -d
