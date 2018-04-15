
def get_path(*arg):
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
