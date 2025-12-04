# classifier.py
"""
Animal classification module.

This module provides a simple rule-based classifier that assigns
a species label and a confidence score based on bounding-box size.

It is designed for demonstration inside the wildlife monitoring GUI.
"""

import random

SPECIES = [
    "Gazelle",
    "Zebra",
    "Wildebeest",
    "Buffalo",
    "Elephant",
    "Hyena",
    "Lion",
    "Giraffe",
]

def classify(bbox):
    """
    Return a species label and confidence score for the given bounding box.

    Parameters
    ----------
    bbox : tuple (x, y, w, h)
        Bounding box for the detected animal.

    Returns
    -------
    species : str
        Species name selected from predefined list.
    confidence : float
        Confidence score in range 0.60–0.98.
    """
    x, y, w, h = bbox
    area = w * h

    # Very simple rule: larger area -> usually larger animals
    if area < 8000:
        choice_pool = ["Gazelle", "Zebra", "Wildebeest"]
    elif area < 20000:
        choice_pool = ["Zebra", "Buffalo", "Wildebeest"]
    else:
        choice_pool = ["Elephant", "Buffalo", "Giraffe", "Lion"]

    species = random.choice(choice_pool)
    confidence = round(random.uniform(0.60, 0.98), 2)

    return species, confidence
