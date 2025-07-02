#!/bin/bash
# Create a null sink named "assistant_output"
pactl load-module module-null-sink sink_name=assistant_output sink_properties=device.description="AssistantOutput"
# Find the default source (microphone) and sink (speakers)
DEFAULT_SINK=$(pactl get-default-sink)
# Loopback the assistant's output to the default speakers
pactl load-module module-loopback source=assistant_output.monitor sink="$DEFAULT_SINK"
echo "Audio setup complete."
