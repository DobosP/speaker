import sounddevice as sd

print("--- Detailed Audio Device Check ---")
try:
    devices = sd.query_devices()
    if not devices:
        print("!!! No audio devices found. !!!")
    else:
        print("Available audio devices:")
        for i, device in enumerate(devices):
            print(f"Device {i}:")
            print(f"  Name: {device['name']}")
            print(f"  Host API: {sd.query_hostapis(device['hostapi'])['name']}")
            print(f"  Max Input Channels: {device['max_input_channels']}")
            print(f"  Max Output Channels: {device['max_output_channels']}")
            print(f"  Default Samplerate: {device['default_samplerate']}")
except Exception as e:
    print(f"An error occurred while querying devices: {e}")
print("--- Check finished ---")