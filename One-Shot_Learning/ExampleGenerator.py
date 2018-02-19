
from PIL import Image
import os
import numpy as np
import random

# Return random subfolder of given path.
def rnd_subfolder(path):
    x = os.listdir(path)
    return random.choice(x)

# Pick another letter in the same alphabet.
def pick_other_letter(path, alphabet, letter, version, similar_letter):
    letter2 = letter
    version2 = version

    if similar_letter:
        # Pick random similar letter.
        while (True):
            version2 = rnd_subfolder(path + "/" + alphabet + "/" + letter2)
            if version2 != version:
                break
    else:
        # Pick random non-similar letter.
        while (True):
            letter2 = rnd_subfolder(path + "/" + alphabet)
            version2 = rnd_subfolder(path + "/" + alphabet + "/" + letter2)
            if letter2 != letter and version2 != version:
                break

    return letter2, version2

# Get random examples. (1:1 ratio between true and false).
def load_examples(path, count, verbose = False, include_meta_data = False):
    counter = 0
    # Init example array.
    examples = []

    # Randomize the initial example target.
    initial_target = random.randint(0, 1)

    # Generate and add examples to list of examples.
    while counter < count:
        similar_letter = counter % 2 == initial_target

        # Pick random alphabet, letter and version.
        alphabet = rnd_subfolder(path)
        letter = rnd_subfolder(path + "/" + alphabet)
        version = rnd_subfolder(path + "/" + alphabet + "/" + letter)

        # Pick other letter from the same alphabet.
        letter2, version2 = pick_other_letter(path, alphabet, letter, version, similar_letter)

        # Add generated example to list of examples.
        new_example = [
            np.array(Image.open(path + "/" + alphabet + "/" + letter + "/" + version).getdata()),
            np.array(Image.open(path + "/" + alphabet + "/" + letter2 + "/" + version2).getdata()),
            similar_letter
        ]
        if include_meta_data:
            new_example.append(path + "/" + alphabet + "/" + letter + "/" + version)
            new_example.append(path + "/" + alphabet + "/" + letter2 + "/" + version2)
        examples.append(new_example)

        # Print loading information if "verbose".
        if verbose and (counter % 100 == 0 or counter == 0):
            print("loading: " + str(counter * 100 / count) + "%")

        counter += 1

    # Shuffling is needed due to fixed target order.
    random.shuffle(examples)
    return examples
