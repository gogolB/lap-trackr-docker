#!/bin/bash

echo Restarting Cameras...
sudo systemctl restart nvargus-daemon

sudo systemctl restart zed_x_daemon

echo Done!
