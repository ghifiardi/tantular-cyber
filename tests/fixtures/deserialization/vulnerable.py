import pickle
import pickle as pk
from pickle import loads

import cPickle
import yaml


def load_blob(raw):
    return pickle.loads(raw)


def load_file(handle):
    return pickle.load(handle)


def load_legacy(raw):
    return cPickle.loads(raw)


def parse_config(text):
    return yaml.load(text)


def parse_unsafe(text):
    return yaml.unsafe_load(text)


def load_aliased_name(raw):
    return loads(raw)


def load_aliased_module(raw):
    return pk.loads(raw)
