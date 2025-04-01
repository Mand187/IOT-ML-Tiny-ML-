import serial
from PIL import Image
import numpy as np
import time
import os

# Serial port settings
port = 6  # Change to your COM port
baud = 115200
ser = serial.Serial(f'COM{port}', baud, timeout=1)

# Image dimensions
WIDTH, HEIGHT = 176, 144
RESIZED_WIDTH, RESIZED_HEIGHT = 64, 64

# Output folder
OUTPUT_FOLDER = "images"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def wait_for_string(target_string, timeout=5):
    buffer = ""
    start_time = time.time()
    while time.time() - start_time < timeout:
        if ser.in_waiting > 0:
            char = ser.read(1).decode(errors="ignore")
            buffer += char
            if target_string in buffer:
                print(f"Found: '{target_string}'")
                return True
            if len(buffer) > 100:
                buffer = buffer[-50:]
    print(f"Timeout waiting for '{target_string}'")
    return False

def receive_image():
    print("Waiting for image data...")
    image_data = bytearray()

    # Wait for START marker
    if not wait_for_string("START", 5):
        raise RuntimeError("Did not receive START marker")

    # Read image data
    while len(image_data) < WIDTH * HEIGHT:
        chunk = ser.read(WIDTH * HEIGHT - len(image_data))
        if not chunk:
            raise RuntimeError("Image receive timeout")
        image_data.extend(chunk)

    # Verify END marker
    if not wait_for_string("END", 5):
        raise RuntimeError("Did not receive END marker")

    # Convert to image
    data = np.frombuffer(image_data, dtype=np.uint8).reshape((HEIGHT, WIDTH))
    return Image.fromarray(data, "L")

def main():
    if not wait_for_string("Arduino Ready"):
        print("Arduino did not initialize properly")
        return

    while True:
        try:
            custom_name = "Box"
            
            # Receive image from Arduino
            original_img = receive_image()
            
            # Process the image (resize)
            processed_img = original_img.resize((RESIZED_WIDTH, RESIZED_HEIGHT))
            
            # Generate filename
            if not custom_name:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                custom_name = f"image_{timestamp}"
            
            original_path = f"{OUTPUT_FOLDER}/{custom_name}_original.png"
            processed_path = f"{OUTPUT_FOLDER}/{custom_name}_processed.png"
            
            # Save images
            original_img.save(original_path)
            processed_img.save(processed_path)
            
            print(f"Saved images: {original_path}, {processed_path}")
            
            # Small delay before next image
            time.sleep(0.5)

        except RuntimeError as e:
            print(f"Error: {e}")
            time.sleep(1)  # Retry delay

if __name__ == "__main__":
    main()
