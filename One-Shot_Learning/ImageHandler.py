
from PIL import Image
import math


def image_differences_2D(img1, img2):
    """
    Generates an image that represents the differences between two images.

    :param img1: 2D list representing an image.
    :param img2: 2D list representing an image.
    :return: 2D list representing an image.
    """
    img_diff = []
    for i, r in enumerate(img1):
        new_row = []
        for j, c in enumerate(r):
            if c == img2[i][j]:
                new_row.append(0)
            else:
                new_row.append(1)
        img_diff.append(new_row)
    return img_diff


def normalize_image(image):
    """
    Normalize image between 0 and 1.

    :param image: 1D list representing an image.
    :return: 1D list representing an image.
    """
    for i, pixel in enumerate(image):
        if pixel != 0:
            image[i] = 1
    return image


def load_image_raw(path):
    """
    Load an image given a path.

    :param path: path to an image:
    :return: 1D list representing an image.
    """
    image = list(Image.open(path).getdata())
    return normalize_image(image)


def load_image_compressed(path, size):
    """
    Load an image and compress it given a path.

    :param path: path to an image:
    :param size: desired size of the compressed image.
    :return: 1D list representing an image.
    """
    image = Image.open(path)
    image = image.resize((size, size), Image.ANTIALIAS)
    image = list(image.getdata())
    return normalize_image(image)


def get_object_frame(image):
    """
    Calculate object frame.

    :param image: a 2D list representing an image.
    :return: a 4-tuple representing the frame of the object.
      (x_min, x_max, y_min, y_max)
    """
    x_min = len(image) + 1
    x_max = -1
    y_min = len(image) + 1
    y_max = -1

    for i, r in enumerate(image):
        for j, c in enumerate(r):
            if c == 0:
                if i < y_min: y_min = i
                if i > y_max: y_max = i
                if j < x_min: x_min = j
                if j > x_max: x_max = j

    return x_min, x_max, y_min, y_max


def crop_image(image, x_min, x_max, y_min, y_max):
    """
    Crops an image given a cropping box within th image.

    :param image: a 2D list representing an image.
    :param x_min, x_max, y_min, y_max: representing the box to be cropped.
    :return: 2D list representing the cropped image.
    """
    new_image = []
    for i, r in enumerate(image):
        new_row = []
        for j, c in enumerate(r):
            if i >= y_min and i <= y_max and j >= x_min and j <= x_max:
                new_row.append(c)
        if len(new_row) > 0:
            new_image.append(new_row)
    return new_image


def image_2D_to_1D(image):
    """
    Convert a 2D list into a 1D list.

    :param image: 2D list.
    :return: 1D list.
    """
    new_image = []
    for r in image:
        for c in r:
            new_image.append(c)
    return new_image


def image_1D_to_2D(image):
    """
    Converts a 1D list to 2D list (squared matrix).

    :param image: 1D list.
    :return: 2D list.
    """
    size = int(math.sqrt(len(image)))
    new_image = []
    r = []
    for i, pixel in enumerate(image):
        r.append(pixel)
        if (i + 1) % size == 0:
            new_image.append(r)
            r = []
    return new_image


def resize_2D_image(image, size):
    """
    Resize a 2D image to a desired squared size.

    :param image: a 2D list representing an image.
    :param size: the desired size of the image.
    :return: 2D list representing the resized image).
    """
    image_flattened = image_2D_to_1D(image)
    image_zoomed = Image.new(mode="L", size=(int(len(image[0])), int(len(image))))
    image_zoomed.putdata(image_flattened)
    image_scaled = image_zoomed.resize((size, size))
    return image_1D_to_2D(list(image_scaled.getdata()))


def extract_visual_object_2D(image):
    """
    Recognizes an object in a 2D list, crops the image, and resize it.

    :param image: a 2D list representing an image.
    :return: a 2D list representing the object.
    """
    x_min, x_max, y_min, y_max = get_object_frame(image)
    image_cropped = crop_image(image, x_min, x_max, y_min, y_max)
    return resize_2D_image(image_cropped, size=len(image))


def extract_visual_object_1D(image):
    """
    Recognizes an object in a 1D list, crops the image, and resize it.

    :param image: a 1D list representing an image.
    :return: a 1D list representing the object.
    """
    image_2D = image_1D_to_2D(image)
    image_object = extract_visual_object_2D(image_2D)
    return image_2D_to_1D(image_object)
