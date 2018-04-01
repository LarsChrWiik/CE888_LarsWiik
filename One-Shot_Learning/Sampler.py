
from PIL import Image
import os
import numpy as np
import random


"""
Shuffle two lists accordingly. 
"""
def shuffle_two_lists(l1, l2):
    z = list(zip(l1, l2))
    random.shuffle(z)
    a, b = zip(*z)
    return list(a), list(b)


"""
Get column from matrix. 
"""
def column(matrix, i):
    return [row[i] for row in matrix]


"""
Class containing static methods for feature sector samples
for character recognition. 
Each sample contains 2*105*105 pixels with 
"""
class Sampler:

    """
    Get random subfolder of given path.
    """
    @staticmethod
    def __rnd_subfolder(path, alphabet=None, character=None):
        full_path = path
        if alphabet is not None:
            full_path += "/" + alphabet
            if character is not None:
                full_path += "/" + character
        x = os.listdir(full_path)
        x = [x for x in x if x != ".DS_Store"]
        return random.choice(x)


    """
    Load raw image as a list given path.
    Contain numbers from 0 to 255. 
    """
    @staticmethod
    def load_image_raw(path, alphabet, character, version):
        return list(Image.open(path + "/" + alphabet + "/" + character + "/" + version).getdata())


    """
    Load image as a list given path and compress it.
    Contain numbers from 0 to 255. 
    """
    @staticmethod
    def load_image_compressed(path, alphabet, character, version, size):
        image = Image.open(path + "/" + alphabet + "/" + character + "/" + version)
        image_compressed = image.resize((size, size), Image.ANTIALIAS)
        return list(image_compressed.getdata())


    """
    Load image as a list given path.
    """
    @staticmethod
    def load_image(path, alphabet, character, version):
        image = Sampler.load_image_raw(
            path=path,
            alphabet=alphabet,
            character=character,
            version=version
        )

        # Transform the image into input features between 0 and 1.
        for i, pixel in enumerate(image):
            if pixel != 0:
                image[i] = 1

        return image


    """
    Generate path for new random sample. 
    """
    @staticmethod
    def __get_rnd_sample(path):
        # Pick random alphabet, character, and version.
        alphabet = Sampler.__rnd_subfolder(path)
        character = Sampler.__rnd_subfolder(path, alphabet)
        version = Sampler.__rnd_subfolder(path, alphabet, character)
        return alphabet, character, version


    """
    Generate path for random character in same alphabet.
    """
    @staticmethod
    def __pick_other_character(path, alphabet, character, version, same_character):
        character2 = character
        version2 = version

        if same_character:
            # Pick random same character.
            while True:
                version2 = Sampler.__rnd_subfolder(path, alphabet, character2)
                if version2 != version:
                    break
        else:
            # Pick random non-same character.
            while True:
                character2 = Sampler.__rnd_subfolder(path, alphabet)
                version2 = Sampler.__rnd_subfolder(path, alphabet, character2)
                if character2 != character and version2 != version:
                    break

        return character2, version2


    """
    Generate one sample. 
    """
    @staticmethod
    def __get_sample(path, same_character):

        # Pick random alphabet, character, and version.
        alphabet, character, version = Sampler.__get_rnd_sample(path=path)
        image1 = Sampler.load_image(path, alphabet, character, version)

        # Pick other character from the same alphabet. (either same or not).
        character2, version2 = Sampler.__pick_other_character(
            path, alphabet, character, version, same_character
        )
        image2 = Sampler.load_image(path, alphabet, character2, version2)

        Y = 0 if same_character else 1

        return image1, image2, Y


    """
    Generate several samples with equal target ratio.
    One sample contains one feature vector X with two images with a corresponding target Y. 
    """
    @staticmethod
    def get_samples(path, count):
        # Generate random initial target.
        same_character = random.randint(0, 1) == 0
        samples = []
        counter = 0
        while counter < count:
            # Generate sample.
            image1, image2, Y = Sampler.__get_sample(
                path=path,
                same_character=same_character
            )
            X = image1 + image2
            sample = [X, Y]
            samples.append(sample)
            same_character = False if same_character else True
            counter += 1

        # Concatenate columns.
        X = column(samples, 0)
        Y = column(samples, 1)
        X, Y = shuffle_two_lists(X, Y)
        return X, Y


    """
    Generate a sample and convert it to a 2D array. 
    """
    @staticmethod
    def get_samples_2D(path, count):
        X, Y = Sampler.get_samples(path=path, count=count)
        X_new = []
        for sample in X:
            new_sample = []
            r = []
            for i, pixel in enumerate(sample):
                r.append(pixel)
                if (i+1) % 105 == 0:
                    new_sample.append(r)
                    r = []
            X_new.append(new_sample)
        return X_new, Y


    """
    Implementation of N-way one-shot learning generation. 
    """
    @staticmethod
    def n_way_one_shot_learning(path, n=20):

        # Pick initial Image.
        alphabet, character, version = Sampler.__get_rnd_sample(path)
        image_main = Sampler.load_image(path, alphabet, character, version)
        X = []
        Y = []

        # Pick other character from the same alphabet.
        character2, version2 = Sampler.__pick_other_character(
            path, alphabet, character, version, same_character=True
        )
        similar_character = Sampler.load_image(path, alphabet, character2, version2)
        X.append(similar_character)
        Y.append(0) # 1 indicates: Similar character.

        # Add n-1 non similar images.
        for i in range(n-1):
            # Pick other character from the same alphabet.
            c, v = Sampler.__pick_other_character(
                path, alphabet, character, version, same_character=False
            )
            non_similar_character = Sampler.load_image(path, alphabet, c, v)
            X.append(non_similar_character)
            Y.append(1) # 1 indicates: Not similar character.

        X, Y = shuffle_two_lists(X, Y)

        return image_main, X, Y
