# Use the official image as a parent image
FROM node:current-slim

# Set the working directory
WORKDIR /usr/src/app

# Copy the files
COPY model/ /model/
RUN ls -la /model/*

# Installing stuff
RUN apt-get update
RUN apt-get install -y python3 python3-pip
RUN pip3 install imageio opencv-python
RUN pip3 install validators numpy
RUN pip3 install scikit-image
RUN pip3 install flask
RUN pip3 install requests
RUN pip3 install pathlib
RUN apt-get install -y libsm6 libxext6 libxrender-dev

RUN pip3 install --upgrade pip
RUN apt update
RUN pip3 install --upgrade tensorflow==2.1.0

# Inform Docker that the container is listening on the specified port at runtime.
EXPOSE 5000

# Copy the rest of your app's source code from your host to your image filesystem.
COPY . .

ENTRYPOINT [ "python3" ]
CMD [ "predict.py" ]

