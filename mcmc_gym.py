import numpy as np
import pandas as pd
import numpy.random as rnd
from matplotlib import animation
import pymc3 as pm


def train_logistic_model(data, features, target, step):
    model_dict = dict()

    # Create dictionary for all important features & target
    for feat in features:
        model_dict[feat] = data[feat]
    model_dict[target] = data[target]

    # Convert to dataframe to be used by pymc3
    model_df = pd.DataFrame(model_dict)
    with pm.Model() as logistic_model:

        # Create equation given features & target
        equation = target + " ~ " + " + ".join(features)

        # Construct logistic model with Binomial RV target and given equation
        pm.glm.GLM.from_formula(equation, model_df, family=pm.glm.families.Binomial())
        trace = pm.sample(400, step=step)

        return trace


def plot_traces(traces, burnin=200):
    '''
    Convenience function:
    Plot traces with overlaid means and values
    '''

    ax = pm.traceplot(traces[burnin:], figsize=(12, len(traces.varnames) * 1.5),
                      lines={k: v['mean'] for k, v in pm.summary(traces[burnin:]).iterrows()})

    for i, mn in enumerate(pm.summary(traces[burnin:])['mean']):
        ax[i, 0].annotate('{:.2f}'.format(mn), xy=(mn, 0), xycoords='data'
                          , xytext=(5, 10), textcoords='offset points', rotation=90
                          , va='bottom', fontsize='large', color='#AA0022')

if __name__ == '__main__':
    data = pd.read_csv("adult_us_postprocessed.csv")
    features = data.columns[:-1]
    target = data.columns[-1]
    # run metropolis-hastings
    trace = train_logistic_model(data, features, target, pm.Metropolis())
    plot_traces(trace)