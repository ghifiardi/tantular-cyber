"""Benchmark corpus case (synthetic). DO NOT RUN OR DEPLOY. Labels live in manifest.json."""

import pickle

import yaml


def restore_job(raw_bytes):
    job = pickle.loads(raw_bytes)
    return job["status"]


def read_settings(text):
    return yaml.load(text)


def read_settings_safe(text):
    return yaml.safe_load(text)


def read_trusted_looking(text):
    return yaml.unsafe_load(text)
