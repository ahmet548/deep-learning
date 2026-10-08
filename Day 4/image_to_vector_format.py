from PIL import Image
import numpy as np

img = Image.open('ai.png')
vector = np.array(img)

print(vector)

img2 = Image.open('ai.jpg')
vector2 = np.array(img2)

print(vector2)