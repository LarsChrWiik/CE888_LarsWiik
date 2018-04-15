
import sys

"""
Display progressbar. 
"""
def show_progressbar(i, max_i, prefix="", finish=False):
    """
    Displays a progressbar.

    :param i: int, iteration number.
    :param max_i: int, maximum iteration number.
    :param prefix: string, any words before the score is displayed.
    :param finish: bool, indicating if this is the last message of the progress bar.
    """
    sys.stdout.write("\r")
    sys.stdout.write(prefix + str(i * 100 / max_i) + "%")
    sys.stdout.flush()
    if finish: print("")
