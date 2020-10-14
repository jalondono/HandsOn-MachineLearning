import yaml
import zipfile
import pandas as pd


def load_yalm():
    """
    load the hyper params from a yml file
    :param args: dictionary with the data
    :return: a dictionary with hyper params
    """
    try:
        with open("parameters.yml", "r") as f:
            params = yaml.safe_load(f)
        return params['classifier'], params['NN']
    except FileNotFoundError:
        classifier = {'epochs': 300,
                      'batch_size': 125,
                      'shuffle': True,
                      'verbose': False},
        model_params = {'lr': 0.005,
                        'l1': 0,
                        'epochs': 80,
                        'batch': 32,
                        'layers': [256, 128]}
        return classifier, model_params


def load_data(zipfolder, filename, cols):
    """
    Get the data from a zip file
    :param cols:
    :param filename:
    :param zipfolder:
    :return: train dataset
    """
    with zipfile.ZipFile(zipfolder, 'r') as zip_ref:
        zip_ref.extractall()
    if cols is None:
        data = pd.read_csv(filename)
    else:
        data = pd.read_csv(filename, usecols=cols)
    return data
