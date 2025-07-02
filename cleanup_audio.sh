#!/bin/bash
# Unload the loopback module
pactl unload-module module-loopback
# Unload the null sink module
pactl unload-module module-null-sink
echo "Audio cleanup complete."
