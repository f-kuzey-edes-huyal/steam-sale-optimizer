FROM apache/airflow:2.9.0-python3.10

# Switch to root to install system dependencies
USER root

RUN apt-get update && apt-get install -y libgomp1

# Install dependencies needed for Chrome and Selenium
RUN apt-get update && apt-get install -y \
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
    --no-install-recommends && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install Google Chrome (stable channel)
RUN wget -q -O - https://dl.google.com/linux/linux_signing_key.pub | gpg --dearmor > /usr/share/keyrings/google.gpg && \
    echo "deb [arch=amd64 signed-by=/usr/share/keyrings/google.gpg] http://dl.google.com/linux/chrome/deb/ stable main" > /etc/apt/sources.list.d/google-chrome.list && \
    apt-get update && \
    apt-get install -y google-chrome-stable && \
    rm -rf /var/lib/apt/lists/*

# Install matching ChromeDriver version 137.0.7151.103
ENV CHROMEDRIVER_VERSION=137.0.7151.103
RUN wget -O /tmp/chromedriver_linux64.zip https://edgedl.me.gvt1.com/edgedl/chrome/chrome-for-testing/${CHROMEDRIVER_VERSION}/linux64/chromedriver-linux64.zip && \
    unzip /tmp/chromedriver_linux64.zip -d /tmp/ && \
    mv /tmp/chromedriver-linux64/chromedriver /usr/local/bin/chromedriver && \
    chmod +x /usr/local/bin/chromedriver && \
    rm -rf /tmp/chromedriver_linux64.zip /tmp/chromedriver-linux64

# Ensure PATH includes ChromeDriver
ENV PATH="/usr/local/bin:$PATH"
ENV CHROME_BIN="/usr/bin/google-chrome"
ENV CHROMEDRIVER_BIN="/usr/local/bin/chromedriver"

# Copy requirements.txt
COPY requirements.txt /requirements.txt

# Switch to airflow user before installing Python packages
USER airflow

# Install Python packages with version pins for Flask/Connexion compatibility
RUN pip install --no-cache-dir --timeout=1000 --retries=10 \
    flask==2.2.5 \
    connexion[swagger-ui]==2.14.2 && \
    pip install --no-cache-dir --timeout=1000 --retries=10 -r /requirements.txt
