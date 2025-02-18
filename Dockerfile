FROM python:3.11-slim
ENV PYTHONUNBUFFERED 1
RUN mkdir /app
COPY . /app
WORKDIR /app
RUN pip3 install --upgrade pip
RUN pip3 install pipenv
RUN pipenv install --system
RUN pip install tensorflow
CMD gunicorn --bind 0.0.0.0:$PORT dermaApi.wsgi
EXPOSE 8000
# remember to buid the image with this tag --platform linux/amd64