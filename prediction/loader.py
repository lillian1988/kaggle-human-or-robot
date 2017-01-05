# encoding: UTF-8

import pandas as pd


def read_bids_data():
    return pd.read_csv("data/bids.csv.zip", header=0)


def read_bidders_data():
    return pd.read_csv("data/train.csv.zip", header=0)


def read_testset():
    return pd.read_csv("data/test.csv.zip", header=0)
