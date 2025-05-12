from pathlib import Path
from statistics import mean

import numpy as np
import pandas as pd


def make_coord_list_collection_for_plot(csv_list: list, column: int):
    coord_list_collection = []

    for i in range(len(csv_list)):
        coord_list_collection.append([])

        irc_data_list = np.loadtxt(
            csv_list[i], skiprows=1, delimiter=",", encoding="UTF8"
        )

        coord_list_collection[i] = irc_data_list[:, column].tolist()

    return coord_list_collection


def unit_conversion_for_energy_list_collection(energy_list_collection: list, unit: str):
    def unit_conversion_for_energy_list(energy_list: list, unit: str):
        def hartree_to_kj_per_mol(hartree: float):
            return hartree * 2625.500

        def hartree_to_kcal_per_mol(hartree: float):
            return hartree * 627.5095

        if unit == "kj_per_mol":
            converted_energy_list = list(map(hartree_to_kj_per_mol, energy_list))

        elif unit == "kcal_per_mol":
            converted_energy_list = list(map(hartree_to_kcal_per_mol, energy_list))

        else:
            converted_energy_list = energy_list[:]
            print("Unit conversion failed")

        return converted_energy_list

    converted_energy_list_collection = []

    for index, _ in enumerate(energy_list_collection):
        converted_energy_list = unit_conversion_for_energy_list(
            energy_list=energy_list_collection[index], unit=unit
        )
        converted_energy_list_collection.append(converted_energy_list)

    return converted_energy_list_collection


def make_lower_energy_ts_irc_index_list(energy_list_collection: list, ncollect: int):
    irc_index_and_ts_energy_for_df = {"irc_index": [], "ts_energy": []}

    for index, energy_list in enumerate(energy_list_collection):
        irc_index_and_ts_energy_for_df["irc_index"].append(index)
        ts_energy = max(energy_list)
        irc_index_and_ts_energy_for_df["ts_energy"].append(ts_energy)

    irc_index_and_ts_energy_df = pd.DataFrame(irc_index_and_ts_energy_for_df)
    sorted_irc_index_and_ts_energy_df = irc_index_and_ts_energy_df.sort_values(
        by="ts_energy"
    )

    target_irc_index_list = (
        sorted_irc_index_and_ts_energy_df["irc_index"].iloc[:ncollect].tolist()
    )

    return target_irc_index_list


def collect_irc_data_list_by_index_list(
    irc_data_list_collection: list, index_list: list
):
    collected_irc_data_list_collection = []

    for index in index_list:
        collected_irc_data_list_collection.append(irc_data_list_collection[index])

    return collected_irc_data_list_collection


def calc_ts_energy_avg_from_irc_energy_list_collection(energy_list_collection: list):
    ts_energy_list = []

    for energy_list in energy_list_collection:
        ts_energy = max(energy_list)
        ts_energy_list.append(ts_energy)

    ts_energy_avg = mean(ts_energy_list)

    return ts_energy_avg


def clean_energy_list_collection(energy_list_collection: list, reference_energy: float):
    def clean_energy_list(energy_list: list, reference_energy: float):
        def clean_energy(energy, reference_energy):
            return energy - reference_energy

        reference_energy_list = []

        for _ in energy_list:
            reference_energy_list.append(reference_energy)

        cleaned_energy_list = list(
            map(clean_energy, energy_list, reference_energy_list)
        )

        return cleaned_energy_list

    cleaned_energy_list_collection = []

    for index, _ in enumerate(energy_list_collection):
        cleaned_energy_list = clean_energy_list(
            energy_list=energy_list_collection[index], reference_energy=reference_energy
        )
        cleaned_energy_list_collection.append(cleaned_energy_list)

    return cleaned_energy_list_collection


def make_ts_data_list(data_list_collection: list, energy_list_collection: list):
    ts_data_list = []

    for index, energy_list in enumerate(energy_list_collection):
        ts_index = energy_list.index(max(energy_list))
        ts_data = data_list_collection[index][ts_index]
        ts_data_list.append(ts_data)

    return ts_data_list


def plot_irc_of_previous_and_one_wn(
    save_path: str,
    gas_ts1_r1: list,
    gas_ts1_r2: list,
    gas_ts1_energy: list,
    gas_ts1_reaction_path_length: list,
    gas_ts1_ts_r1: list,
    gas_ts1_ts_r2: list,
):
    pass


def main():
    # Input section
    save_path = "./irc_plot_of_previous_and_one_wn"
    plot_type1 = "plot_irc_path_via_two_bonds"
    plot_type2 = "plot_irc_energy_along_reaction_path_length"
    gas_irc_data_csv_collection_path = "./gas/gas_irc_data_csv_collection"
    pcm_irc_data_csv_collection_path = "./pcm/pcm_irc_data_csv_collection"
    w45_irc_data_csv_collection_path = "./w45/new_w45_irc_data_csv_collection_path"
    ncollect = 20

    # Data processing
    gas_ts1_csv_list = list(
        Path(f"{gas_irc_data_csv_collection_path}/ts1").glob("*.csv")
    )
    gas_ts2_csv_list = list(
        Path(f"{gas_irc_data_csv_collection_path}/ts2").glob("*.csv")
    )
    pcm_ts1_csv_list = list(
        Path(f"{pcm_irc_data_csv_collection_path}/ts1").glob("*.csv")
    )
    pcm_ts2_csv_list = list(
        Path(f"{pcm_irc_data_csv_collection_path}/ts2").glob("*.csv")
    )
    w45_ts1_csv_list = list(
        Path(f"{pcm_irc_data_csv_collection_path}/ts1").glob("*.csv")
    )
    w45_ts2_csv_list = list(
        Path(f"{pcm_irc_data_csv_collection_path}/ts2").glob("*.csv")
    )

    gas_ts1_r1 = make_coord_list_collection_for_plot(
        csv_list=gas_ts1_csv_list, column=0
    )
    gas_ts1_r2 = make_coord_list_collection_for_plot(
        csv_list=gas_ts1_csv_list, column=1
    )
    gas_ts1_energy = make_coord_list_collection_for_plot(
        csv_list=gas_ts1_csv_list, column=2
    )
    gas_ts1_reaction_path_length = make_coord_list_collection_for_plot(
        csv_list=gas_ts1_csv_list, column=3
    )

    gas_ts2_r1 = make_coord_list_collection_for_plot(
        csv_list=gas_ts2_csv_list, column=0
    )
    gas_ts2_r2 = make_coord_list_collection_for_plot(
        csv_list=gas_ts2_csv_list, column=1
    )
    gas_ts2_energy = make_coord_list_collection_for_plot(
        csv_list=gas_ts2_csv_list, column=2
    )
    gas_ts2_reaction_path_length = make_coord_list_collection_for_plot(
        csv_list=gas_ts2_csv_list, column=3
    )

    pcm_ts1_r1 = make_coord_list_collection_for_plot(
        csv_list=pcm_ts1_csv_list, column=0
    )
    pcm_ts1_r2 = make_coord_list_collection_for_plot(
        csv_list=pcm_ts1_csv_list, column=1
    )
    pcm_ts1_energy = make_coord_list_collection_for_plot(
        csv_list=pcm_ts1_csv_list, column=2
    )
    pcm_ts1_reaction_path_length = make_coord_list_collection_for_plot(
        csv_list=pcm_ts1_csv_list, column=3
    )

    pcm_ts2_r1 = make_coord_list_collection_for_plot(
        csv_list=pcm_ts2_csv_list, column=0
    )
    pcm_ts2_r2 = make_coord_list_collection_for_plot(
        csv_list=pcm_ts2_csv_list, column=1
    )
    pcm_ts2_energy = make_coord_list_collection_for_plot(
        csv_list=pcm_ts2_csv_list, column=2
    )
    pcm_ts2_reaction_path_length = make_coord_list_collection_for_plot(
        csv_list=pcm_ts2_csv_list, column=3
    )

    w45_ts1_r1 = make_coord_list_collection_for_plot(
        csv_list=w45_ts1_csv_list, column=0
    )
    w45_ts1_r2 = make_coord_list_collection_for_plot(
        csv_list=w45_ts1_csv_list, column=1
    )
    w45_ts1_energy = make_coord_list_collection_for_plot(
        csv_list=w45_ts1_csv_list, column=2
    )
    w45_ts1_reaction_path_length = make_coord_list_collection_for_plot(
        csv_list=w45_ts1_csv_list, column=3
    )

    w45_ts2_r1 = make_coord_list_collection_for_plot(
        csv_list=w45_ts2_csv_list, column=0
    )
    w45_ts2_r2 = make_coord_list_collection_for_plot(
        csv_list=w45_ts2_csv_list, column=1
    )
    w45_ts2_energy = make_coord_list_collection_for_plot(
        csv_list=w45_ts2_csv_list, column=2
    )
    w45_ts2_reaction_path_length = make_coord_list_collection_for_plot(
        csv_list=w45_ts2_csv_list, column=3
    )

    gas_ts1_energy = unit_conversion_for_energy_list_collection(
        energy_list_collection=gas_ts1_energy, unit="kcal_per_mol"
    )
    gas_ts2_energy = unit_conversion_for_energy_list_collection(
        energy_list_collection=gas_ts2_energy, unit="kcal_per_mol"
    )
    pcm_ts1_energy = unit_conversion_for_energy_list_collection(
        energy_list_collection=pcm_ts1_energy, unit="kcal_per_mol"
    )
    pcm_ts2_energy = unit_conversion_for_energy_list_collection(
        energy_list_collection=pcm_ts2_energy, unit="kcal_per_mol"
    )
    w45_ts1_energy = unit_conversion_for_energy_list_collection(
        energy_list_collection=w45_ts1_energy, unit="kcal_per_mol"
    )
    w45_ts2_energy = unit_conversion_for_energy_list_collection(
        energy_list_collection=w45_ts2_energy, unit="kcal_per_mol"
    )

    target_w45_ts1_irc_index_list = make_lower_energy_ts_irc_index_list(
        energy_list_collection=w45_ts1_energy, ncollect=ncollect
    )
    target_w45_ts2_irc_index_list = make_lower_energy_ts_irc_index_list(
        energy_list_collection=w45_ts2_energy, ncollect=ncollect
    )

    w45_ts1_r1 = collect_irc_data_list_by_index_list(
        irc_data_list_collection=w45_ts1_r1, index_list=target_w45_ts1_irc_index_list
    )
    w45_ts1_r2 = collect_irc_data_list_by_index_list(
        irc_data_list_collection=w45_ts1_r2, index_list=target_w45_ts1_irc_index_list
    )
    w45_ts1_energy = collect_irc_data_list_by_index_list(
        irc_data_list_collection=w45_ts1_energy,
        index_list=target_w45_ts1_irc_index_list,
    )
    w45_ts1_reaction_path_length = collect_irc_data_list_by_index_list(
        irc_data_list_collection=w45_ts1_reaction_path_length,
        index_list=target_w45_ts1_irc_index_list,
    )

    w45_ts2_r1 = collect_irc_data_list_by_index_list(
        irc_data_list_collection=w45_ts2_r1, index_list=target_w45_ts2_irc_index_list
    )
    w45_ts2_r2 = collect_irc_data_list_by_index_list(
        irc_data_list_collection=w45_ts2_r2, index_list=target_w45_ts2_irc_index_list
    )
    w45_ts2_energy = collect_irc_data_list_by_index_list(
        irc_data_list_collection=w45_ts2_energy,
        index_list=target_w45_ts2_irc_index_list,
    )
    w45_ts2_reaction_path_length = collect_irc_data_list_by_index_list(
        irc_data_list_collection=w45_ts2_reaction_path_length,
        index_list=target_w45_ts2_irc_index_list,
    )

    gas_ts1_energy_max = max(gas_ts1_energy[0])
    w45_ts1_energy_avg = calc_ts_energy_avg_from_irc_energy_list_collection(
        energy_list_collection=w45_ts1_energy
    )

    gas_ts1_energy = clean_energy_list_collection(
        energy_list_collection=gas_ts1_energy, reference_energy=gas_ts1_energy_max
    )
    gas_ts2_energy = clean_energy_list_collection(
        energy_list_collection=gas_ts2_energy, reference_energy=gas_ts1_energy_max
    )
    pcm_ts1_energy = clean_energy_list_collection(
        energy_list_collection=pcm_ts1_energy, reference_energy=gas_ts1_energy_max
    )
    pcm_ts2_energy = clean_energy_list_collection(
        energy_list_collection=pcm_ts2_energy, reference_energy=gas_ts1_energy_max
    )
    w45_ts1_energy = clean_energy_list_collection(
        energy_list_collection=w45_ts1_energy, reference_energy=w45_ts1_energy_avg
    )
    w45_ts2_energy = clean_energy_list_collection(
        energy_list_collection=w45_ts2_energy, reference_energy=w45_ts1_energy_avg
    )

    gas_ts1_ts_r1 = make_ts_data_list(
        data_list_collection=gas_ts1_r1, energy_list_collection=gas_ts1_energy
    )
    gas_ts1_ts_r2 = make_ts_data_list(
        data_list_collection=gas_ts1_r2, energy_list_collection=gas_ts1_energy
    )
    gas_ts2_ts_r1 = make_ts_data_list(
        data_list_collection=gas_ts2_r1, energy_list_collection=gas_ts2_energy
    )
    gas_ts2_ts_r2_ = make_ts_data_list(
        data_list_collection=gas_ts2_r2, energy_list_collection=gas_ts2_energy
    )
    pcm_ts1_ts_r1 = make_ts_data_list(
        data_list_collection=pcm_ts1_r1, energy_list_collection=pcm_ts1_energy
    )
    pcm_ts1_ts_r2 = make_ts_data_list(
        data_list_collection=pcm_ts1_r2, energy_list_collection=pcm_ts1_energy
    )
    pcm_ts2_ts_r1 = make_ts_data_list(
        data_list_collection=pcm_ts2_r1, energy_list_collection=pcm_ts2_energy
    )
    pcm_ts2_ts_r2 = make_ts_data_list(
        data_list_collection=pcm_ts2_r2, energy_list_collection=pcm_ts2_energy
    )
    w45_ts1_ts_r1 = make_ts_data_list(
        data_list_collection=w45_ts1_r1, energy_list_collection=w45_ts1_energy
    )
    w45_ts1_ts_r2 = make_ts_data_list(
        data_list_collection=w45_ts1_r2, energy_list_collection=w45_ts1_energy
    )
    w45_ts2_ts_r1 = make_ts_data_list(
        data_list_collection=w45_ts2_r1, energy_list_collection=w45_ts2_energy
    )
    w45_ts2_ts_r2 = make_ts_data_list(
        data_list_collection=w45_ts2_r2, energy_list_collection=w45_ts2_energy
    )


if __name__ == "__main__":
    main()
