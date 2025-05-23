import copy
import tomllib
from pathlib import Path
from statistics import mean

import pandas as pd


def imput_toml_read(input_toml: Path) -> dict:
    with open(input_toml, "rb") as f:
        input_data = tomllib.load(f)

    return input_data


def make_irc_csv_path_obj_dict_for_ptsb(irc_csv_path_dict: dict) -> dict:
    irc_csv_path_obj_dict = {}

    for key in irc_csv_path_dict:
        irc_csv_path_obj_dict[key]["ts1"] = Path(f"{irc_csv_path_dict[key]}/ts1").glob(
            "*.csv"
        )
        irc_csv_path_obj_dict[key]["ts2"] = Path(f"{irc_csv_path_dict[key]}/ts2").glob(
            "*.csv"
        )

    return irc_csv_path_obj_dict


def make_irc_data_dict_for_ptsb(irc_csv_path_obj_dict: dict):
    irc_data_dict = {}

    for key in irc_csv_path_obj_dict:
        irc_data_dict[key] = {}

        for ts_type in irc_csv_path_obj_dict[key]:
            irc_data_dict[key][ts_type] = {}

            for irc_csv_path_obj in irc_csv_path_obj_dict[key]:
                irc_data_df = pd.read_csv(irc_csv_path_obj)
                csv_stem = irc_csv_path_obj.stem

                for column in irc_data_df.columns.values:
                    if column not in irc_data_dict[key]:
                        irc_data_dict[key][ts_type][column] = {}

                    irc_data_dict[key][ts_type][column][csv_stem] = irc_data_df[
                        column
                    ].tolist()

        return irc_data_dict


def classify_ircs_by_product_for_ptsb(
    irc_data_dict: dict, classification_option_dict: dict
) -> dict:
    classified_irc_data_dict = {}

    for key in irc_data_dict:
        if classification_option_dict["general"][key]:
            classified_irc_data_dict[key] = copy.deepcopy(irc_data_dict)

        else:
            classified_irc_data_dict[key] = {}

            for ts_type in irc_data_dict[key]:
                if classification_option_dict["ts_type"][ts_type]:
                    classified_irc_data_dict[key][ts_type] = copy.deepcopy(
                        irc_data_dict[key][ts_type]
                    )

                else:
                    classified_irc_data_dict[key][ts_type] = {}

                    for prod_type in classification_option_dict[
                        "prod_judge_param"
                    ].values():
                        classified_irc_data_dict[key][ts_type][prod_type] = {}

                    for prod_type in classified_irc_data_dict[key][ts_type]:
                        prod_judge_param_name = classification_option_dict[
                            "prod_judge_param_name"
                        ][prod_type]
                        prod_judge_criterion = classification_option_dict[
                            "prod_judge_criteria"
                        ][prod_judge_param_name]

                        target_csv_stem_list = []

                        for csv_stem, prod_judge_param_list in irc_data_dict[key][
                            ts_type
                        ][prod_judge_param_name].items():
                            prod_judge_param = prod_judge_param_list[-1]

                            if prod_judge_param < prod_judge_criterion:
                                target_csv_stem_list.append(csv_stem)

                        for param_name, param_dict in irc_data_dict[key][
                            ts_type
                        ].items():
                            classified_irc_data_dict[key][ts_type][prod_type][
                                param_name
                            ] = {csv_stem: param_dict[csv_stem] in target_csv_stem_list}

    return classified_irc_data_dict


def exclude_ircs_eith_high_energy_ts():
    pass


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
    # 1. Input section

    # 1.1. Read input
    input_file = Path("./input.toml")
    input_data = imput_toml_read(input_toml=input_file)

    # 1.2. General input
    save_path = input_data["general"]["save_path"]
    ncollect = input_data["general"]["ncollect"]

    # 1.3. CSV collection path input
    irc_data_csv_collection_path_dict = input_data["csv_collection_path"]

    # 1.4. Color input
    plot_color_dict = input_data["plot_color"]

    # 1.5. Classification option input
    classification_option_dict = input_data["classification_option"]

    # 2. Data processing section

    # 2.1. Make IRC data csv path object dictionary
    irc_data_csv_path_obj_dict = make_irc_csv_path_obj_dict_for_ptsb(
        irc_csv_path_dict=irc_data_csv_collection_path_dict
    )

    # 2.2. Make IRC data dict
    irc_data_dict = make_irc_data_dict_for_ptsb(
        csv_path_obj_dict=irc_data_csv_path_obj_dict
    )

    # 2.3. Classify IRCs by product
    classified_irc_data_dict = classify_ircs_by_product_for_ptsb(
        irc_data_dict=irc_data_dict,
        classification_option_dict=classification_option_dict,
    )

    # 2.4. Collect IRCs with low-energy transition states

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


# 3. Plot section

if __name__ == "__main__":
    main()
