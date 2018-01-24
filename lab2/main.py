
import bootstrap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random


# ***** FUNTIONS *****

def save_diagram(plotter, filename, x_label = None, y_label = None):
    axes = plt.gca()
    if not x_label == None:
        axes.set_xlabel(x_label)
    if not y_label == None:
        axes.set_ylabel(y_label)
    plotter.savefig("Diagrams/" + filename + ".png")
    plotter.savefig("Diagrams/" + filename + ".pdf")

def create_diagrams():
    # Scaterplot "current fleet" and "proposed fleet".
    sns_plot = sns.lmplot(
        df.columns[0],
        df.columns[1],
        data = df,
        fit_reg = False
    )
    save_diagram(sns_plot, 'scaterplot')

    # Histogram "current fleet"
    plt.clf()
    histogram_plot = sns.distplot(
        df.values.T[0],
        bins = 20,
        kde = False
    ).get_figure()
    save_diagram(
        histogram_plot,
        'histogram_current_fleet',
        'MPG',
        'Number of vehicles'
    )

    # Histogram "proposed fleet"
    plt.clf()
    histogram_plot = sns.distplot(
        df.dropna().values.T[1],
        bins = 20,
        kde = False
    ).get_figure()
    save_diagram(
        histogram_plot,
        'histogram_proposed_fleet',
        'MPG',
        'Number of vehicles'
    )

# ***** CODE *****

df = pd.read_csv('vehicles.csv')

# Creating Scaterplot and Histogram of fleets.
create_diagrams()

# ***** Standard deviation comparison via the boostrap *****
#std_current_fleet = np.std(df.values.T[0])
#std_proposed_fleet = np.std(df.dropna().values.T[0])
print("## Current fleet")
print("")
print("# Standard deviation")
boot = bootstrap.boostrap(np.std, 10000, df.values.T[0])
print("- upper = " + str(boot[2]))
print("- std-mean = " + str(boot[0]))
print("- lower = " + str(boot[1]))
print("")
print("Mean:")
print("- mean = " + str(np.mean(df.values.T[0])))
print("")

print("## Proposed fleet")
print("")
print("# Standard deviation:")
boot = bootstrap.boostrap(np.std, 10000, df.dropna().values.T[1])
print("- upper = " + str(boot[2]))
print("- std-mean = " + str(boot[0]))
print("- lower = " + str(boot[1]))
print("")
print("Mean:")
print("- mean = " + str(np.mean(df.dropna().values.T[1])))
print("")


#print(("Mean: %f") % (np.mean(data)))
#print(("Median: %f") % (np.median(data)))
#print(("Var: %f") % (np.var(data)))
#print(("std: %f") % (np.std(data)))
#print(("MAD: %f") % (mad(data)))
