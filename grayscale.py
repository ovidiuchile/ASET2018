import numpy as np
from PIL import Image

np.set_printoptions(threshold=np.nan)

# image = sys.argv[1]
img = Image.open('test_image.jpg').convert('L')
img.save('test_image_gray.jpg')


arr = 255 - np.array(img)
print(arr.shape)
#print(arr)
