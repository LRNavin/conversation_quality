#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: load_participants.py
# Created Date: Wednesday, May 22nd 2019, 2:05:34 pm
# Author: Chirag Raman
#
# Copyright (c) 2019 Chirag Raman
###

import argparse

from pathlib import Path
import numpy as np
import pandas as pd

import constants

def load_participants(path):
    """ Load the dictionary of participants - {day : array(participant_id)}

    Args:
        path    : The path to the participants csv

    Returns:
        The {day : array(participant_id)} dictionary
    """
    p = pd.read_csv(path, header=None)
    participants = {}
    for day in p[0].unique():
        participants[day] = p[p[0]==day][1].values
    return participants


def load_annotations(path, lost_path, participants_path):
    """ Load the manual annotations and add hierarchical indices

    Args:
        path                : The path to the annotations csv
        lost_path           : The path to the lost csv denoting participants
                              outside the camera FOV
        participants_path   : The path to the participants id

    Returns:
        The annotations data frame with a hierarchical column index
        ["day", "participant", "annotation"] where "annotation" corresponds
        to the nine types of labels
    """
    # Read all required csvs
    annotations = pd.read_csv(path, header=None)
    participants = pd.read_csv(participants_path, header=None)
    lost = pd.read_csv(lost_path, header=None)

    # Construct and set index
    days = np.repeat(participants[0].values, constants.N_LABELS)
    ps = np.repeat(participants[1].values, constants.N_LABELS)
    ls = np.tile(constants.LABELS, len(participants))
    index = pd.MultiIndex.from_arrays(
        [days, ps, ls],
        names=("day", "participant", "annotation")
    )
    annotations.columns = index

    # Set entries of lost participants to -1
    repeated_lost = pd.DataFrame(
        np.repeat(lost.values, constants.N_LABELS, axis=1)
    )
    mask = (repeated_lost[repeated_lost.columns] == 1)
    mask.columns = annotations.columns
    annotations[mask] = -1

    return annotations


def load_F_formations(root_dir):
    """ Load the F-formation annotations into a dataframe

    Args:
        root_dir    : The directory containing the day-wise csv's

    Returns:
        The dataframe containing annotations from all days. The index of the
        dataframe is an added column "day" with values "day<1/2/3>. Annotations
        for a certain day (eg. Day 1) can then be accessed with
        `df.loc["day1",:]`

    """
    day_annotations = []
    for day_file in root_dir.glob("*.csv"):
        day_anno = pd.read_csv(day_file)
        day_anno["day"] = day_file.stem.lower()
        day_anno.set_index("day", inplace=True)
        day_annotations.append(day_anno)
    concat_annotations = pd.concat(day_annotations)
    return concat_annotations


def main():

    extract_fform=False
    extract_sacts=True

    """ Load and serialize the labels and F-formations data frames """
    outdir = constants.main_dataset_storage
    # Make the output directory
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if extract_sacts:
        # Load labels
        print("Loading Annotations")
        annotations = load_annotations(constants.labels_annot, constants.lost_annot, constants.participant_annot)
        annotations.to_pickle((outdir / "annotations.pickle"))

    if extract_fform:
        # Load F-formations
        print("Loading F-formations")
        f_formations = load_F_formations(constants.fform_annot_data)
        f_formations.to_pickle((outdir / "f_formations.pickle"))


if __name__ == "__main__":
    main()