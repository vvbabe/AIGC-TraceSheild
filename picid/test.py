#!/usr/bin/env python
# -*- coding: utf-8 -*-

from stegastamp import add_watermark, extract_watermark

add_watermark("input.png", "GPT", "output.png")
msg = extract_watermark("output.png")
print("id:", msg)
