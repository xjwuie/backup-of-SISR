from PIL import Image
import scipy.misc
import numpy as np


def upsample(pic_file, scale, output_file, mode='bicubic'):
    img = scipy.misc.imread(pic_file)
    x, y, z = img.shape
    img = scipy.misc.imresize(img, (x * scale, y * scale), mode)
    img = np.array(img, np.uint8)
    img = np.clip(img, 0, 255)

    img = Image.fromarray(img, 'RGB')
    img.save(output_file)


if __name__ == '__main__':
    img_file = 'bird.jpg'
    output_file = 'bird_bic.jpg'
    upsample(img_file, 3, output_file)



