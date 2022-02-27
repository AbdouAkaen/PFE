import argparse
import base64
from datetime import datetime
import os
import shutil
import numpy as np
import socketio
import eventlet
import eventlet.wsgi
from PIL import Image
from flask import Flask
from io import BytesIO
import torch
from torch.autograd import Variable
from network_model import model_cnn
import utils

#initialize our server
sio = socketio.Server()

#our flask (web) app
app = Flask(__name__)

#init our model and image array as empty
model = None
prev_image_array = None

#set min/max speed for our autonomous car
MAX_SPEED = 10
MIN_SPEED = 3

#and a speed limit
speed_limit = MAX_SPEED


#registering event handler for the server
@sio.on('telemetry')
def telemetry(sid, data):
    if data:

        # The current steering angle of the car
        st_ang = data["steering_angle"].replace(",",".")
        th = data["throttle"].replace(",",".")
        sp = data["speed"].replace(",",".")
        steering_angle = float(st_ang)
        # The current throttle of the car, how hard to push peddle
        throttle = float(th)
        # The current speed of the car
        speed = float(sp)

        # The current image from the center camera of the car
        image = Image.open(BytesIO(base64.b64decode(data["image"])))
        img = np.asarray(image)
        img = utils.preprocess(img)
        try:

            # predict the steering angle for the image
            img = Variable(torch.cuda.FloatTensor([img])).permute(0,3,1,2)

            steering_angle_throttle = model(img)
            steering_angle = steering_angle_throttle.item()

            # make sure we slow down first and then go back to the original max speed.
            global speed_limit
            if speed > speed_limit:
                speed_limit = MIN_SPEED  # slow down
            else:
                speed_limit = MAX_SPEED
            throttle = 1 - steering_angle**2 - (speed/speed_limit)**2

            print('sterring_angle: {} throttle: {} spped: {}'.format(steering_angle, throttle, speed))
            send_control(steering_angle, throttle)
        except Exception as e:
            print(e)

        # save frame
        if args.image_folder != '':
            timestamp = datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]
            image_filename = os.path.join(args.image_folder, timestamp)
            image.save('{}.jpg'.format(image_filename))
    else:

        sio.emit('manual', data={}, skip_sid=True)

# Function to connect and communiucate with simulator socket.
@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)

# Send the generated steering angle and acceleration value to the simulator 
def send_control(steering_angle, throttle):
    sio.emit(
        "steer",
        data={
            'steering_angle': steering_angle.__str__().replace(".",","),
            'throttle': throttle.__str__().replace(".",",")
        },
        skip_sid=True)
    print("send control from client")

# The main function to run the autonomous mode of the vehicle.
if __name__ == '__main__':
    print('1 =======================')
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument(
        'model_weights',
        type=str,
        help='Path to model h5 file. Model should be on the same path.'
    )
    parser.add_argument(
        'image_folder',
        type=str,
        nargs='?',
        default='',
        help='Path to image folder. This is where the images from the run will be saved.'
    )
    args = parser.parse_args()

    #load model
    print('2 =======================')

    model = model_cnn().cuda()
    model.load_state_dict(torch.load(args.model_weights))
    model.eval()

    print('3 =======================')

    if args.image_folder != '':
        print("Creating image folder at {}".format(args.image_folder))
        if not os.path.exists(args.image_folder):
            os.makedirs(args.image_folder)
        else:
            shutil.rmtree(args.image_folder)
            os.makedirs(args.image_folder)
        print("RECORDING THIS RUN ...")
    else:
        print("NOT RECORDING THIS RUN ...")

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)
    print('4 =======================')

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
    
    print('5 =======================')

    
    
