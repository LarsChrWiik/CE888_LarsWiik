
import ImageHandler
import os
import random


def load_image(path, compress=True, compression_size=28):
    """
    Load image given path.

    :param path: string
    :param compress: bool representing whether compression should be done.
    :param compression_size: int representing compression size for a squared image.
    :return:
    """
    if compress:
        return ImageHandler.load_image_compressed(path=path, size=compression_size)
    return ImageHandler.load_image_raw(path=path)


def get_sample(dataset, same_character):
    """
    Generate an image sample.

    :param dataset: string representing the dataset name.
    :param same_character: bool representing if the the new image should be similar or not.
    :return: 2-tuple containing a list of two images X and a list of the target Y.
       (X, Y)
    """
    alphabet, character, version = __get_rnd_sample(dataset=dataset)
    image1 = load_image(path=__get_path(dataset, alphabet, character, version))
    character2, version2 = __pick_other_character(dataset, alphabet, character, version, same_character)
    image2 = load_image(path=__get_path(dataset, alphabet, character2, version2))
    Y = 0 if same_character else 1
    X = [image1, image2]
    return X, Y


def generate_samples(dataset, count):
    """
    Generate a desired count of samples. The ratio between targets is balanced.

    :param dataset:
    :param count:
    :return: 2-tuple containing one list of 1D lists of samples X and a 1D list of targets Y.
       (X, Y)
    """
    same_character = random.randint(0, 1) == 0
    samples = []
    counter = 0
    while counter < count:
        X_single, Y_single = get_sample(dataset=dataset, same_character=same_character)
        samples.append([X_single, Y_single])
        same_character = False if same_character else True
        counter += 1
    X = __column(samples, 0)
    Y = __column(samples, 1)
    X, Y = __shuffle_two_lists(X, Y)
    return X, Y


def n_way_one_shot_learning(dataset, n=20):
    """
    Sampling of N-way one-shot learning test.

    :param dataset: string representing the dataset name.
    :param n: the number of images to compare with.
    :return: 3-tuple representing the main image, n images, and n targets.
       (image_main, X, Y)
    """
    # Pick initial Image.
    alphabet, character, version = __get_rnd_sample(dataset)
    image_main = load_image(path=__get_path(dataset, alphabet, character, version))

    # Generate another version of the main image.
    character2, version2 = __pick_other_character(
        dataset, alphabet, character, version, same_character=True
    )
    image_similar = load_image(path=__get_path(dataset, alphabet, character2, version2))

    # Initialize the samples X with according targets Y.
    X = [image_similar]
    Y = [0]

    # Add n-1 images of other characters.
    for i in range(n-1):
        c, v = __pick_other_character(dataset, alphabet, character, version, same_character=False)
        image_non_similar = load_image(path=__get_path(dataset, alphabet, c, v))
        X.append(image_non_similar)
        Y.append(1)

    X, Y = __shuffle_two_lists(X, Y)

    return image_main, X, Y


"""
                 PRIVATE FUNCTIONS
"""


def __get_path(*arg):
    """
    Construct the path given a variable number of sub-folders.

    :param arg: list represent the sub-folders and the last file
    :return: string representing the final path.
    """
    path = ""
    for i, folder in enumerate(arg):
        path += folder
        if i != len(arg)-1:
            path += "/"
    return path


def __shuffle_two_lists(l1, l2):
    """
    Shuffle two lists accordingly.

    :param l1: list
    :param l2: list
    :return: list that is shuffled
    """
    z = list(zip(l1, l2))
    random.shuffle(z)
    a, b = zip(*z)
    return list(a), list(b)


def __column(matrix, i):
    """
    Get column from matrix.

    :param matrix: 2D list
    :param i: int that represent the column index that should be extracted.
    :return: list representing the desired column from the matrix.
    """
    return [row[i] for row in matrix]


def __rnd_subfolder(path):
    """
    Get random subfolder of given path.

    :param path: string.
    :return: string representing a path of a random sub-folder of the given path.
    """
    x = os.listdir(path)
    x = [x for x in x if x != ".DS_Store"]
    return random.choice(x)


def __get_rnd_sample(dataset):
    """
    Generate path for new random sample.

    :param dataset: string representing the dataset name.
    :return: 3-tuple containing alphabet, character, and version names.
    """
    # Pick random alphabet, character, and version.
    alphabet = __rnd_subfolder(dataset)
    character = __rnd_subfolder(__get_path(dataset, alphabet))
    version = __rnd_subfolder(__get_path(dataset, alphabet, character))
    return alphabet, character, version


def __pick_other_character(dataset, alphabet, character, version, same_character):
    """
    Generate path for a random character in same alphabet.

    :param dataset: string representing the dataset name.
    :param alphabet: string representing the alphabet name.
    :param character: string representing the character name.
    :param version: string representing the version name.
    :param same_character: bool representing if the the new image should be similar or not.
    :return: 2-tuple containing strings representing the new character and version.
    """
    character2 = character
    version2 = version

    if same_character:
        # Pick random same character.
        while True:
            version2 = __rnd_subfolder(__get_path(dataset, alphabet, character2))
            if version2 != version:
                break
    else:
        # Pick random non-same character.
        while True:
            character2 = __rnd_subfolder(__get_path(dataset, alphabet))
            version2 = __rnd_subfolder(__get_path(dataset, alphabet, character2))
            if character2 != character and version2 != version:
                break

    return character2, version2
