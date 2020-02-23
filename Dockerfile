# Use the official image as a parent image
FROM node:current-slim

# Set the working directory
WORKDIR /usr/src/app

# Copy the file from your host to your current location
COPY model/ /model/
RUN ls -la /model/*

# Run the command inside your image filesystem
# RUN npm install
RUN apt-get update
RUN apt-get -y install curl
RUN apt-get -y install gnupg gnupg1 gnupg2
RUN echo "deb [arch=amd64] http://storage.googleapis.com/tensorflow-serving-apt stable tensorflow-model-server tensorflow-model-server-universal" | tee /etc/apt/sources.list.d/tensorflow-serving.list && curl https://storage.googleapis.com/tensorflow-serving-apt/tensorflow-serving.release.pub.gpg | tac | tac | apt-key add -
RUN apt-get update
RUN apt-get install tensorflow-model-server
RUN tensorflow_model_server --version

# Inform Docker that the container is listening on the specified port at runtime.
EXPOSE 9000

# Run the specified command within the container.
CMD [ "tensorflow_model_server", "--model_base_path=/usr/src/app/model/",  "--rest_api_port=9000", "--model_name=Resnet"]

# Copy the rest of your app's source code from your host to your image filesystem.
COPY . .
