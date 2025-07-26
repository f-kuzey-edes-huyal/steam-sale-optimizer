

cd terraform

terraform init

terraform apply (you will be asked docker_username , docker_password , admin_password )

This line will create a public_ip_adress like public_ip_address = "172.190.247.168" 

Enter the virtual machine using this ip adress: 


ssh username@172.190.247.168 (you will be asked for enter admin password )

# copy .env file to azure vm write this code in local environment while you ere inside stem-sale=optimizer

scp .env azureuser@20.185.222.163:/home/azureuser/.env

# now in azure machine you have to write the code below to delete older docker version:

 sudo mv /usr/bin/docker-compose /usr/bin/docker-compose.old

# then

 sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose

sudo chmod +x /usr/local/bin/docker-compose

 /usr/local/bin/docker-compose --version (you will see an upgraded version)

 cd /opt/steam-sale-optimizer

# move the .env file 

sudo mv /home/azureuser/.env /opt/steam-sale-optimizer/.env

# to avoid docker error
sudo usermod -aG docker azureuser
exit
ssh azureuser@20.185.222.163

cd /opt/steam-sale-optimizer
docker-compose -f docker-compose.azure.yml up -d --build
# after this step i got errors related credentials. the steps that are given below solved my issue


1. Completely remove or rename all Docker credential helpers
sudo mv /usr/bin/docker-credential-secretservice /usr/bin/docker-credential-secretservice.bak 2>/dev/null
sudo mv /usr/bin/docker-credential-pass /usr/bin/docker-credential-pass.bak 2>/dev/null
sudo mv /usr/local/bin/docker-credential-secretservice /usr/local/bin/docker-credential-secretservice.bak 2>/dev/null
sudo mv /usr/local/bin/docker-credential-pass /usr/local/bin/docker-credential-pass.bak 2>/dev/null

2. Create a truly empty Docker config directory:
sudo mkdir -p /tmp/docker-config
sudo bash -c 'echo "{}" > /tmp/docker-config/config.json'
sudo chmod 644 /tmp/docker-config/config.json

sudo DOCKER_CONFIG=/tmp/docker-config docker-compose -f /opt/steam-sale-optimizer/docker-compose.yml up -d --build

# 🚀 Containers Successfully Constructed in Cloud

![Containers Constructed](https://github.com/f-kuzey-edes-huyal/steam-sale-optimizer/blob/main/figures/terraform_resources.png)

---

# 🌐 Check the IP Address Given with Virtual Machine

![VM IP Address](https://github.com/f-kuzey-edes-huyal/steam-sale-optimizer/blob/main/figures/containers_in_azure_vm.png)

---

# ⚙️ Check the Link of FastAPI IP Address (Same as the Virtual Machine)

The FastAPI app runs on the same IP address as the virtual machine. Use this IP to access the app.

![FastAPI with Terraform](https://github.com/f-kuzey-edes-huyal/steam-sale-optimizer/blob/main/figures/fastapi_container_with_terraform.png)



