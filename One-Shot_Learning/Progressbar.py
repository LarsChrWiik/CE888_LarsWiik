
import sys

"""
Display progressbar. 
"""
def show_progressbar(i, max_i, prefix="", finish=False):
    sys.stdout.write("\r")
    sys.stdout.write(prefix + str(i * 100 / max_i) + "%")
    sys.stdout.flush()
    if finish: print("")
