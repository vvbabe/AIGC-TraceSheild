import argparse
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
    print(f"Bits to be embedded: {bits}")

    h, w, _ = pixels.shape

    if len(bits) > h * w:
        raise ValueError("Message too long for image.")

    flat = pixels.reshape(-1, 3)
    for i, bit in enumerate(bits):
        flat[i][0] = (flat[i][0] & ~1) | bit  # encode into LSB of Red channel

    encoded_pixels = flat.reshape((h, w, 3))
    encoded_img = Image.fromarray(encoded_pixels.astype(np.uint8))
    encoded_img.save(output_path, "PNG")
    print(f"Watermarked image saved to: {output_path}")


def extract_watermark(image_path, length=10):  # Changed default length to 10
    img = Image.open(image_path).convert("RGB")
    pixels = np.array(img).reshape(-1, 3)
    bits = [(px[0] & 1) for px in pixels[:length * 8]]
    print(f"Extracted bits: {bits}")  # Print the extracted bits
    return bits_to_str(bits)


def main():
    parser = argparse.ArgumentParser(description="Embed and extract watermarks in images.")
    subparsers = parser.add_subparsers(dest="command")

    # Add watermark subcommand
    add_parser = subparsers.add_parser("add", help="Add watermark to an image")
    add_parser.add_argument("image_path", help="Path to the input image")
    add_parser.add_argument("text", help="Text to embed as watermark")
    add_parser.add_argument("output_path", help="Path to save the output image")

    # Extract watermark subcommand
    extract_parser = subparsers.add_parser("extract", help="Extract watermark from an image")
    extract_parser.add_argument("image_path", help="Path to the input image")
    # Removed --length argument as it's no longer needed

    args = parser.parse_args()

    if args.command == "add":
        add_watermark(args.image_path, args.text, args.output_path)
    elif args.command == "extract":
        message = extract_watermark(args.image_path)  # Default length is 10 now
        print("Extracted watermark message:", message)

if __name__ == "__main__":
    main()
