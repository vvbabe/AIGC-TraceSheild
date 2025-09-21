# stegastamp.py
from PIL import Image
import numpy as np

def str_to_bits(s):
    return [int(b) for char in s for b in format(ord(char), '08b')]

def bits_to_str(bits):
    chars = [chr(int("".join(map(str, bits[i:i+8])), 2)) for i in range(0, len(bits), 8)]
    return ''.join(chars)

def add_watermark(image_path, text, output_path):
    img = Image.open(image_path).convert("RGB")
    pixels = np.array(img)
    
    print(f"Converting text to bits: {text}")
    bits = str_to_bits(text)
    print(f"Bits to be embedded: {bits}")  # Check the bits

    h, w, _ = pixels.shape

    if len(bits) > h * w:
        raise ValueError("Message too long for image.")

    flat = pixels.reshape(-1, 3)
    for i, bit in enumerate(bits):
        flat[i][0] = (flat[i][0] & ~1) | bit  # encode into LSB of Red channel

    encoded_pixels = flat.reshape((h, w, 3))
    encoded_img = Image.fromarray(encoded_pixels.astype(np.uint8))
    encoded_img.save(output_path, "PNG")


def extract_watermark(image_path, length=3):
    img = Image.open(image_path).convert("RGB")
    pixels = np.array(img).reshape(-1, 3)
    bits = [(px[0] & 1) for px in pixels[:length * 8]]
    print("Extracted bits:", bits)  # Print the extracted bits
    return bits_to_str(bits)
