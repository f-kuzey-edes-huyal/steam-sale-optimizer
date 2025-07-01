FROM apache/airflow:2.9.0-python3.10

USER root

# Retry function for shell commands
SHELL ["/bin/bash", "-c"]

# Retry apt-get update & install with 3 attempts
RUN for i in {1..3}; do \
      apt-get update && apt-get install -y libgomp1 && break || \
      (echo "apt-get libgomp1 install failed, retrying... ($i)" && sleep 5); \
    done

RUN for i in {1..3}; do \
      apt-get update && apt-get install -y \
        wget \
        curl \
        unzip \
        gnupg2 \
        libxi6 \
        libgconf-2-4 \
        libnss3 \
        libxss1 \
        libasound2 \
        fonts-liberation \
        libappindicator1 \
        xdg-utils \
        libu2f-udev \
        libvulkan1 \
        --no-install-recommends && break || \
      (echo "apt-get chrome deps install failed, retrying... ($i)" && sleep 5); \
    done && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Google Chrome stable with retry
RUN for i in {1..3}; do \
      wget -q -O - https://dl.google.com/linux/linux_signing_key.pub | gpg --dearmor > /usr/share/keyrings/google.gpg && \
      echo "deb [arch=amd64 signed-by=/usr/share/keyrings/google.gpg] http://dl.google.com/linux/chrome/deb/ stable main" > /etc/apt/sources.list.d/google-chrome.list && \
      apt-get update && apt-get install -y google-chrome-stable && break || \
      (echo "Chrome install failed, retrying... ($i)" && sleep 5); \
    done && \
    rm -rf /var/lib/apt/lists/*

ENV CHROMEDRIVER_VERSION=137.0.7151.103
RUN for i in {1..3}; do \
      wget -O /tmp/chromedriver_linux64.zip https://edgedl.me.gvt1.com/edgedl/chrome/chrome-for-testing/${CHROMEDRIVER_VERSION}/linux64/chromedriver-linux64.zip && \
      unzip /tmp/chromedriver_linux64.zip -d /tmp/ && \
      mv /tmp/chromedriver-linux64/chromedriver /usr/local/bin/chromedriver && \
      chmod +x /usr/local/bin/chromedriver && \
      rm -rf /tmp/chromedriver_linux64.zip /tmp/chromedriver-linux64 && break || \
      (echo "Chromedriver install failed, retrying... ($i)" && sleep 5); \
    done

ENV PATH="/usr/local/bin:$PATH"
ENV CHROME_BIN="/usr/bin/google-chrome"
ENV CHROMEDRIVER_BIN="/usr/local/bin/chromedriver"

COPY requirements.txt /requirements.txt

USER airflow

# Retry pip install with shell loop since pip does not support --retries natively
RUN for i in {1..3}; do \
      pip install --no-cache-dir --timeout=1000 flask==2.2.5 connexion[swagger-ui]==2.14.2 && break || \
      (echo "pip install flask/connexion failed, retrying... ($i)" && sleep 5); \
    done && \
    for i in {1..3}; do \
      pip install --no-cache-dir --timeout=1000 -r /requirements.txt && break || \
      (echo "pip install requirements.txt failed, retrying... ($i)" && sleep 5); \
    done