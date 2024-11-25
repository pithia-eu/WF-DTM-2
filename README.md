# Project Installation and Usage

Follow these steps to set up and run the project:

1. **Set the port number:** Open the file `start_dtm.sh` and change the port number based on the requirement.\
    The command to start the service looks like this:

   `uvicorn main:app --reload --port 8080 --host 0.0.0.0`

2. **Build the Python virtual environment:** 

    Execute the following script `./build_venv.sh`

3. **Enable and start the Rest API:** 

    Execute the following script `./build_service.sh`

4. **Check the service status:** 

    Execute the following command to ensure the service is active `systemctl status dc-dtm.service`

5. **Access the Rest API:** 

    The API can be accessed through the following URL: `http://<your_ip>:<your_port>`

    Remember to replace `<your_ip>` and `<your_port>` with the actual IP and port number you're using.