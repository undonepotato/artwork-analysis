import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import PIL
from io import BytesIO

dataset = pd.read_parquet("data/wikiart/data/")
dataset_mem = dataset.memory_usage(deep=True)  # in bytes
print(dataset_mem)
print(dataset.describe())

# Show an image

img_num = 9  # Change this; if it's 2, for example, it's the second
# image in the dataset.

dict_data = dataset.iloc[img_num, 0]
artist = dataset.iloc[img_num, 1]
dict_img = dict_data["bytes"]

PIL.Image.open(BytesIO(dict_img)).save("test.jpg")

# arr_img = np.asarray(PIL.Image.open(BytesIO(dict_img)))
# imgplot = plt.imshow(arr_img)
# plt.title(artist)
# plt.show()
