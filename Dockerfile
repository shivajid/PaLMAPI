FROM python:3.8-slim-buster

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/shivajid/PaLMAPI.git .


RUN pip3 install -r requirements.txt

EXPOSE 8503

ENTRYPOINT ["streamlit" ,"run", "streamlit_app/TextLLM.py", "--server.port=8503", "--server.address=0.0.0.0"]



