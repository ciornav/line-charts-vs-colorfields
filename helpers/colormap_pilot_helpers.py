import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple
from scipy.stats import tukey_hsd
from scikit_posthocs import posthoc_dunn
from helpers.common_helpers import *
import re
import datetime
import numpy as np

COLORMAPS_COLORFIELDS = ["hi_want_hue", "hmokole", "horanges", "hpurple2",  "hset1", "hyellow_red" ]
COLORMAPS_LINECHARTS = ["lci_want_hue", "lcmokole", "lcset1"]

def task_1_analyzer(df: pd.DataFrame) -> pd.DataFrame:
    # task 1 a empty state + initial state + each reset adds 2 trials
    eq_df = df[df["stepName"] == "task_0"]
    eq_df.reset_index(inplace=True)
    vals = eq_df["stepValue"].to_list()
    new_vals = []
    indexes_to_keep = []
    for index, value in enumerate(vals):
        val = ugly_text_treatment(value)
        item = eval(val)["dataToBeSent"]
        if not are_tasks_understood(item, eq_df.loc[index, "userID"]):
            continue
        indexes_to_keep.append(index)
        item["number_of_trials"] = take_out_resets_and_calculate_moves(item)
        history_data = analyze_timestamps(item)
        new_vals.append({**item, **history_data})
    vals_df = pd.DataFrame(new_vals)
    eq_df_resetted = eq_df.loc[indexes_to_keep, ["userID", "sessionID"]].reset_index(drop=True)
    eq_final = pd.concat([eq_df_resetted, vals_df], axis=1)

    def add_predictions(row):
        try:
            row["electricity_cost"] = row["predictions"]["Model_5"]
        except KeyError:
            row["electricity_cost"] = 1
        return row["electricity_cost"]

    eq_final.dropna(subset=["correctAnswer"], inplace=True)
    eq_final["electricity_cost"] = eq_final.apply(lambda row: add_predictions(row), axis=1)
    history_costs = get_history_highest_and_lowest(eq_final, 0)
    history_df = pd.DataFrame(history_costs)
    eq_final = pd.concat([eq_final, history_df], axis=1)
    return eq_final


def task_2_analyzer(df: pd.DataFrame) -> pd.DataFrame:
    # no empty state here. the initial state is recorded though. Reset takes 2 trials again.
    eq_df = df[df["stepName"] == "task_1"]
    eq_df.reset_index(inplace=True)
    vals = eq_df["stepValue"].to_list()
    new_vals = []
    indexes_to_keep = []
    for index, value in enumerate(vals):
        val = ugly_text_treatment(value)
        item = eval(val)["dataToBeSent"]
        if not are_tasks_understood(item, eq_df.loc[index, "userID"]):
            continue
        indexes_to_keep.append(index)
        item["number_of_trials"] = take_out_resets_and_calculate_moves(item)
        history_data = analyze_timestamps(item)
        new_vals.append({**item, **history_data})
    vals_df = pd.DataFrame(new_vals)
    eq_df_resetted = eq_df.loc[indexes_to_keep, ["userID", "sessionID"]].reset_index(drop=True)
    eq_final = pd.concat([eq_df_resetted, vals_df], axis=1)

    def add_predictions(row):
        row["sum_of_costs"] = np.sum(list(row["predictions"].values()))
        return row["sum_of_costs"]

    eq_final["sum_of_costs"] = eq_final.apply(lambda row: add_predictions(row), axis=1)
    history_costs = get_history_highest_and_lowest(eq_final, 1)
    adjusted_costs = [x if float(x) >= 1.22 else 1.22 for x in history_costs['history_lowest_overall_cost']]
    history_costs['history_lowest_overall_cost'] = adjusted_costs
    history_df = pd.DataFrame(history_costs)
    eq_final = pd.concat([eq_final, history_df], axis=1)
    return eq_final


def task_3_analyzer(df: pd.DataFrame) -> pd.DataFrame:
    # 2 trials for initial state. reset takes one trial.
    eq_df = df[df["stepName"] == "task_2"]
    eq_df.reset_index(inplace=True)
    vals = eq_df["stepValue"].to_list()
    new_vals = []
    indexes_to_keep = []
    for index, value in enumerate(vals):
        val = ugly_text_treatment(value)
        item = eval(val)["dataToBeSent"]
        if not are_tasks_understood(item, eq_df.loc[index, "userID"]):
            continue
        indexes_to_keep.append(index)
        item["number_of_trials"] = take_out_resets_and_calculate_moves(item)
        history_data = analyze_timestamps(item)
        new_vals.append({**item, **history_data})
    vals_df = pd.DataFrame(new_vals)
    eq_df_resetted = eq_df.loc[indexes_to_keep, ["userID", "sessionID"]].reset_index(drop=True)
    eq_final = pd.concat([eq_df_resetted, vals_df], axis=1)

    def add_predictions(row):
        try:
            row["sum_of_costs"] = np.sum(list(row["predictions"].values()))
        except KeyError:
            row["sum_of_costs"] = 5
        return row["sum_of_costs"]

    eq_final["sum_of_costs"] = eq_final.apply(lambda row: add_predictions(row), axis=1)
    history_costs = get_history_highest_and_lowest(eq_final, 2)
    history_df = pd.DataFrame(history_costs)
    eq_final = pd.concat([eq_final, history_df], axis=1)
    return eq_final


def task_4_analyzer(df: pd.DataFrame) -> pd.DataFrame:
    eq_df = df[df["stepName"] == "task_3"]
    eq_df.reset_index(inplace=True)
    vals = eq_df["stepValue"].to_list()
    new_vals = []
    indexes_to_keep = []
    for index, value in enumerate(vals):
        val = ugly_text_treatment(value)
        item = eval(val)["dataToBeSent"]
        if not are_tasks_understood(item, eq_df.loc[index, "userID"]):
            continue
        indexes_to_keep.append(index)
        item["number_of_trials"] = take_out_resets_and_calculate_moves(item)
        history_data = analyze_timestamps(item)
        new_vals.append({**item, **history_data})
    vals_df = pd.DataFrame(new_vals)
    eq_df_resetted = eq_df.loc[indexes_to_keep, ["userID", "sessionID"]].reset_index(drop=True)
    eq_final = pd.concat([eq_df_resetted, vals_df], axis=1)

    def add_predictions(row):
        row["sum_of_costs"] = np.sum(list(row["predictions"].values()))
        return row["sum_of_costs"]

    eq_final["sum_of_costs"] = eq_final.apply(lambda row: add_predictions(row), axis=1)
    history_costs = get_history_highest_and_lowest(eq_final, 3)
    adjusted_costs = [x if float(x) < 0.94 else 0.93 for x in history_costs['history_highest_overall_cost']]
    history_costs['history_highest_overall_cost'] = adjusted_costs
    history_df = pd.DataFrame(history_costs)
    eq_final = pd.concat([eq_final, history_df], axis=1)
    return eq_final


def clean_raw_df(df: pd.DataFrame) -> pd.DataFrame:
    # remove invalid characters from the users' feedback such as //, / and others
    def clean_row(row):
        try:
            space_cleaned = row[4].replace("   ", "")
            row[4] = space_cleaned
            space_cleaned2 = row[3].replace(" ", "")
            row[3] = space_cleaned2
            space_cleaned3 = row[0].replace(" ", "")
            row[0] = space_cleaned3
            space_cleaned4 = row[2].replace("   ", "")
            row[2] = space_cleaned4
            space_cleaned5 = row[1].replace(" ", "")
            row[1] = space_cleaned5
            row[2] = row[2].replace('null', '"null"')
            row[2] = re.sub(
                r'I was looking for sliders that were currently in .*light.* spots but could be moved into .*dark.* spots.',
                'I was looking for sliders that were currently in light spots but could be moved into dark spots.',
                row[2])
            row[2] = re.sub(r'The top slider could be moved .*further.* than the',
                            'The top slider could be moved further than the', row[2])
            row[2] = re.sub(r'repeat until there-s no more adjusting needed. Pay attention to the .*total cost.* bar in the Cost Overview Panel to see if',
                            'repeat until there-s no more adjusting needed. Pay attention to the total cost bar in the Cost Overview Panel to see if', row[2])
            row[2] = re.sub(
                r'I went one slider at the time and compared the previous result to the current one under .*overall cost.*\",\"checkboxesStatuses\"',
                'I went one slider at the time and compared the previous result to the current one under overall cost\" ,\"checkboxesStatuses\"',
                row[2])
            return row
        except AttributeError:
            return row

    df.apply(lambda row: clean_row(row), axis=1)
    return df


def get_invalid_submission_ids() -> dict:
    # this will be only for selecting reiterating combinations.
    # these are users which have non valid submissions due to time, content or early aborting of the study
    users_and_tasks_to_delete = {
        "BD1610C9EB1941958A0D82CE2481392B": [],
    }
    return users_and_tasks_to_delete


def remove_invalid_submissions(df: pd.DataFrame):
    df = df[1:]
    users_to_delete = get_invalid_submission_ids()
    df.reset_index(inplace=True, drop=True)
    indexes_to_remove = []
    for index, val in df["userID"].items():
        if val in users_to_delete.keys():
            indexes_to_remove.append(index)
    good_labels = [i for i in list(df.index) if df.loc[i, "userID"] not in list(users_to_delete.keys())]
    after_user_removal_df = df.iloc[good_labels, :].reset_index()
    return after_user_removal_df


def ugly_text_treatment(value: str) -> str:
    val = value.replace("$$", "")
    val = val.replace('["distance to pareto front for input:', '{"distance to pareto front for input":')
    val = val.replace('with the following prediction:', ',"with the following prediction":')
    val = val.replace('is equal to:', ',"is equal to":')
    val = val.replace('and call index is:', ',"and call index is":')
    val = val.replace('"],"predictions":', '},"predictions":')
    val = val.replace('\\\\', '')
    val = val.replace('false', '"false"')
    val = val.replace('true', '"true"')
    matches = re.findall(r'("and call index is": )(\d+")(\])', val)
    for match in matches:
        original_string = match[0] + match[1] + match[2]
        correct_string = re.sub(r'("and call index is": )(\d+")(\])', r'\1"\2}', original_string)
        val = val.replace(original_string, correct_string)
    return val


def take_out_resets_and_calculate_moves(item: dict, task: int = 0) -> int:
    history = item["history"]
    trials = 0
    if len(history) == 0:
        return trials
    starting_index = 0
    if task == 1 or task == 3:
        initial_state = history[starting_index]["sliderConfig"]
    else:
        starting_index = 1
        try:
            initial_state = history[starting_index]["sliderConfig"]
        except IndexError:
            return trials + 1
    for index, state in enumerate(history):
        if index > starting_index:
            if state["sliderConfig"] == initial_state:
                continue
            else:
                trials += 1
    return trials


def get_history_highest_and_lowest(df: pd.DataFrame, task_number) -> dict:
    #  there is a shift btw slider configs and predictions. Predictions are for the previous config of the slider. Late 1 step.
    history_lowest_overall_cost = []
    history_highest_overall_cost = []
    history_highest_elec_cost = []
    history_lowest_elec_cost = []
    timestamps = []
    for index, el in enumerate(df["history"]):
        user_timestamps = {}
        individual_timestamps = []
        temp_list = []
        elec_list = []
        starting_point = {}
        is_starting_point_reached = False
        for ind, item in enumerate(el):
            if not is_starting_point_reached:
                is_starting_point_reached = isStartingPointReached(item["sliderConfig"], task_number)
                if task_number == 2 and (index == 41 or index == 13):
                    is_starting_point_reached = True
                starting_point = item["sliderConfig"]
                if not is_starting_point_reached:
                    continue
            # if len(item["generDist"]) == 0:
            #     continue
            changes = check_number_of_changes_compared_to_starting_point(starting_point, item["sliderConfig"])
            if task_number in [0, 1, 3] and changes > 1:
                continue
            if len(item["predictions"].values()) == 0:
                continue
            temp_list.append(sum(item["predictions"].values()))
            elec_list.append(item["predictions"]["Model_5"])
            individual_timestamps.append(item["historyTime"])
        user_timestamps[df.iloc[index]["userID"]] = individual_timestamps
        timestamps.append(user_timestamps)
        if len(elec_list) > 0:
            history_highest_elec_cost.append(max(elec_list))
            history_lowest_elec_cost.append(min(elec_list))
        else:
            history_highest_elec_cost.append("nan")
            history_lowest_elec_cost.append("nan")
        if len(temp_list) > 0:
            history_lowest_overall_cost.append(min(temp_list))
            history_highest_overall_cost.append(max(temp_list))
        else:
            history_lowest_overall_cost.append("nan")
            history_highest_overall_cost.append("nan")
    return {
        "history_lowest_overall_cost": history_lowest_overall_cost,
        "history_highest_overall_cost": history_highest_overall_cost,
        "history_highest_elec_cost": history_highest_elec_cost,
        "history_lowest_elec_cost": history_lowest_elec_cost,
    }


def isStartingPointReached(payload, task_number) -> bool:
    starting_point = {}
    if task_number in [0, 1, 2]:
        starting_point = {'x0': 2, 'x1': 3, 'x2': 6, 'x3': 8, 'x4': 10, 'x5': 12, 'x6': 0, 'x7': 0, 'x8': 0, 'x9': 10}
    if task_number == 3:
        starting_point = {'x0': 2, 'x1': 4, 'x2': 5.7, 'x3': 8, 'x4': 9, 'x5': 11.4, 'x6': 11.9, 'x7': 11.2, 'x8': 6.3,
                          'x9': 7}
    if payload == starting_point:
        return True
    else:
        return False


def check_number_of_changes_compared_to_starting_point(starting_point, new_point) -> int:
    number_of_changes = 0
    if list(starting_point.values()) == list(new_point.values()):
        return number_of_changes
    else:
        for start_ind, start_val in enumerate(list(starting_point.values())):
            for new_ind, new_val in enumerate(list(new_point.values())):
                if new_ind != start_ind:
                    continue
                if start_val != new_val:
                    number_of_changes += 1
    return number_of_changes


def analyze_timestamps(item: dict) -> dict:
    try:
        task_start_time = item["history"][0]["historyTime"]
        task_end_time = item["completionTime"]
        end_last_slider_action = item["history"][-1]["historyTime"]
        time_delta = datetime.datetime.strptime(task_end_time, "%Y-%m-%dT%H:%M:%S.%fZ") - datetime.datetime.strptime(
            task_start_time, "%Y-%m-%dT%H:%M:%S.%fZ")
        if len(item["history"]) > 1:
            slider_start = item["history"][1]["historyTime"]
            slider_action_delta = datetime.datetime.strptime(end_last_slider_action,
                                                             "%Y-%m-%dT%H:%M:%S.%fZ") - datetime.datetime.strptime(
                slider_start, "%Y-%m-%dT%H:%M:%S.%fZ")
        else:
            slider_action_delta = datetime.datetime.strptime(end_last_slider_action,
                                                             "%Y-%m-%dT%H:%M:%S.%fZ") - datetime.datetime.strptime(
                task_start_time, "%Y-%m-%dT%H:%M:%S.%fZ")
        start_minus_first = 0
        for index, el in enumerate(item["history"]):
            if index == 0:
                continue
            temp_datetime = datetime.datetime.strptime(el["historyTime"], "%Y-%m-%dT%H:%M:%S.%fZ")
            previous_datetime = datetime.datetime.strptime(item["history"][index - 1]["historyTime"],
                                                           "%Y-%m-%dT%H:%M:%S.%fZ")
            delta = temp_datetime - previous_datetime
            if delta.total_seconds() > start_minus_first:
                start_minus_first = delta.total_seconds()
        return {
            "task_start_time": task_start_time,
            "task_end_time": task_end_time,
            "time_delta_seconds": time_delta.total_seconds(),
            "btw_start_and_first_change": start_minus_first,
            "btw_first_and_last_slider_action": slider_action_delta.total_seconds(),
            "median_viz_loading_time_per_slider_ms": np.median(item["loadingTimePerSld"])
        }
    except IndexError:
        return {
            "task_start_time": 0,
            "task_end_time": 0,
            "time_delta_seconds": 0,
            "btw_start_and_first_change": 0,
            "btw_first_and_last_slider_action": 0,
            "median_viz_loading_time_per_slider_ms": 0
        }


def get_1d_stats_on_colormaps(line_chart_df:pd.DataFrame, heatmap_df: pd.DataFrame, dependent_variable_name: str, task_number: int) -> Tuple[list, list]:
    lc_stats = []
    h_stats = []
    for colormap in line_chart_df["colormap"].unique():
        metric_subset = line_chart_df[line_chart_df["colormap"] == colormap][
            dependent_variable_name].reset_index(drop=True)
        print(
            f"===============generating statistics task {task_number} for line_charts and colormap: {colormap} ==================")
        d = generate_standard_plot_and_stats_for_1d_data(metric_subset, show_plot=False)
        d["name"] = colormap
        lc_stats.append(d)
    for colormap in heatmap_df["colormap"].unique():
        metric_subset = heatmap_df[heatmap_df["colormap"] == colormap][
            dependent_variable_name].reset_index(drop=True)
        print(
            f"===============generating statistics for task {task_number} heatmaps and colormap: {colormap} ==================")
        d = generate_standard_plot_and_stats_for_1d_data(metric_subset, show_plot=False)
        d["name"] = colormap
        h_stats.append(d)
    return lc_stats, h_stats


def data_set_creator(df: pd.DataFrame):
    cvd_df = cvd_analyzer(df)
    eq_df = entry_questionnaire_analyzer(df)
    end_q_df = end_questionnaire_analyzer(df)
    user_config_df = eq_df.loc[:, ["userID", "visualization", "colormap", "scent_height"]]
    t0_df = task_1_analyzer(df)
    t1_df_merged = t0_df.merge(user_config_df, on="userID", how="inner")
    t1_df = task_2_analyzer(df)
    t2_df_merged = t1_df.merge(user_config_df, on="userID", how="inner")
    t2_df = task_3_analyzer(df)
    t3_df_merged = t2_df.merge(user_config_df, on="userID", how="inner")
    t3_df = task_4_analyzer(df)
    t4_df_merged = t3_df.merge(user_config_df, on="userID", how="inner")
    return eq_df, cvd_df, t1_df_merged, t2_df_merged, t3_df_merged, t4_df_merged, end_q_df


def stats_means(df: pd.DataFrame, dependent_variable_name: str) -> dict:
    line_chart_df = df[df["visualization"] == "line_chart"]
    heatmap_df = df[df["visualization"] == "heatmap"]
    lcset1 = line_chart_df[line_chart_df["colormap"] == "set1"][dependent_variable_name].to_list()
    lcmokole = line_chart_df[line_chart_df["colormap"] == "mokole"][dependent_variable_name].to_list()
    lci_want_hue = line_chart_df[line_chart_df["colormap"] == "i_want_hue"][dependent_variable_name].to_list()
    horanges = heatmap_df[heatmap_df["colormap"] == "oranges"][dependent_variable_name].to_list()
    hpurple2 = heatmap_df[heatmap_df["colormap"] == "purple2"][dependent_variable_name].to_list()
    hyellow_red = heatmap_df[heatmap_df["colormap"] == "yellow_red"][dependent_variable_name].to_list()
    hset1 = heatmap_df[heatmap_df["colormap"] == "set1"][dependent_variable_name].to_list()
    hmokole = heatmap_df[heatmap_df["colormap"] == "mokole"][dependent_variable_name].to_list()
    hi_want_hue = heatmap_df[heatmap_df["colormap"] == "i_want_hue"][dependent_variable_name].to_list()
    lc_dfs = [lcset1, lcmokole, lci_want_hue]
    h_dfs = [horanges, hpurple2, hyellow_red, hset1, hmokole, hi_want_hue]
    payload = {}
    lc_normality = True
    for l in lc_dfs:
        if not is_data_normal(l):
            lc_normality = False
    payload["are_lc_populations_normal"] = lc_normality
    h_normality = True
    for h in h_dfs:
        if not is_data_normal(h):
            h_normality = False
    payload["are_h_populations_normal"] = h_normality
    if lc_normality:
        are_lc_means_different = is_the_mean_different(lc_dfs)
        res_lc = tukey_hsd(*lc_dfs)
        payload["tukey_hsd_lc"] = res_lc
    else:
        are_lc_means_different = is_the_mean_different(lc_dfs, method="kruskal")
        res_lc = posthoc_dunn(lc_dfs, p_adjust='holm')
        payload["dunn_lc"] = res_lc
    payload["are_lc_means_different"] = are_lc_means_different
    if h_normality:
        are_h_means_different = is_the_mean_different(h_dfs)
        res_h = tukey_hsd(*h_dfs)
        payload["tukey_hsd_h"] = res_h
    else:
        are_h_means_different = is_the_mean_different(h_dfs, method="kruskal")
        res_h = posthoc_dunn(h_dfs, p_adjust='holm')
        payload["dunn_h"] = res_h
    payload["are_h_means_different"] = are_h_means_different
    return payload


def stats_means_accuracy_colormaps(df: pd.DataFrame, dependent_variable_name: str) -> dict:
    line_chart_df = df[df["visualization"] == "line_chart"]
    heatmap_df = df[df["visualization"] == "heatmap"]
    lcset1 = line_chart_df[line_chart_df["colormap"] == "set1"][dependent_variable_name].to_list()
    lcmokole = line_chart_df[line_chart_df["colormap"] == "mokole"][dependent_variable_name].to_list()
    lci_want_hue = line_chart_df[line_chart_df["colormap"] == "i_want_hue"][dependent_variable_name].to_list()
    horanges = heatmap_df[heatmap_df["colormap"] == "oranges"][dependent_variable_name].to_list()
    hpurple2 = heatmap_df[heatmap_df["colormap"] == "purple2"][dependent_variable_name].to_list()
    hyellow_red = heatmap_df[heatmap_df["colormap"] == "yellow_red"][dependent_variable_name].to_list()
    hset1 = heatmap_df[heatmap_df["colormap"] == "set1"][dependent_variable_name].to_list()
    hmokole = heatmap_df[heatmap_df["colormap"] == "mokole"][dependent_variable_name].to_list()
    hi_want_hue = heatmap_df[heatmap_df["colormap"] == "i_want_hue"][dependent_variable_name].to_list()
    lc_dfs = [lci_want_hue, lcmokole, lcset1]
    h_dfs = [hi_want_hue, hmokole, horanges, hpurple2,  hset1, hyellow_red ]
    payload = {}
    lc_normality = True
    for l in lc_dfs:
        if not is_data_normal(l):
            lc_normality = False
    h_normality = True
    for h in h_dfs:
        if not is_data_normal(h):
            h_normality = False
    resp = get_main_stats(lc_dfs, "lc", COLORMAPS_LINECHARTS, is_parametric=lc_normality)
    payload.update(resp)
    resp = get_main_stats(h_dfs, "h", COLORMAPS_COLORFIELDS, is_parametric=h_normality)
    payload.update(resp)
    return payload


def stats_means_accuracy_viz(df: pd.DataFrame, dependent_variable_name: str) -> dict:
    line_charts = df[df["visualization"] == "line_chart"][dependent_variable_name].to_list()
    heatmaps = df[df["visualization"] == "heatmap"][dependent_variable_name].to_list()
    factor_levels = ["line_charts", "heatmaps"]
    dfs = [line_charts, heatmaps]
    payload = {}
    normality = True
    for l in dfs:
        if not is_data_normal(l):
            normality = False
    payload["are_populations_normal"] = normality
    stats_1d = {}
    for index, d in enumerate(dfs):
        if index == 0:
            print("=====statistics for  line charts. Accuracy vs visualization.========")
            stats_1d["line_charts"] = generate_standard_plot_and_stats_for_1d_data(d, show_plot=False)
        else:
            print("=====statistics for colorfields. Accuracy vs visualization.========")
            stats_1d["colorfields"] = generate_standard_plot_and_stats_for_1d_data(d, show_plot=False)
    payload["stats_1d"] = stats_1d
    resp = get_main_stats(dfs, "viz", factor_levels, is_parametric=normality)
    payload.update(resp)
    return payload


def stats_means_sequential_qualitative_accuracy(df: pd.DataFrame, dependent_variable_name: str) -> dict:
    heatmaps = df[df["visualization"] == "heatmap"]
    horanges = heatmaps[heatmaps["colormap"] == "oranges"]
    hpurple2 = heatmaps[heatmaps["colormap"] == "purple2"]
    hyellow_red = heatmaps[heatmaps["colormap"] == "yellow_red"]
    hset1 = heatmaps[heatmaps["colormap"] == "set1"]
    hmokole = heatmaps[heatmaps["colormap"] == "mokole"]
    hi_want_hue = heatmaps[heatmaps["colormap"] == "i_want_hue"]
    sequentials = pd.concat([horanges, hpurple2, hyellow_red])
    qualitatives = pd.concat([hset1, hmokole, hi_want_hue])
    qualitatives["colormap_type"] = "qualitative"
    sequentials["colormap_type"] = "sequential"
    new_df = pd.concat([qualitatives, sequentials], ignore_index=True)
    new_df.boxplot(column=dependent_variable_name, by=["colormap_type"], showmeans=True)
    plt.ylabel(dependent_variable_name)
    plt.title("colorfields, sequentials vs qualitatives")
    dfs = [sequentials[dependent_variable_name].to_list(), qualitatives[dependent_variable_name].to_list()]
    payload, stats_1d = {}, {}
    normality = True
    for l in dfs:
        if not is_data_normal(l):
            normality = False
    payload["are_populations_normal"] = normality
    for index, d in enumerate(dfs):
        if index == 0:
            print("=====statistics for  sequentials. Accuracy vs visualization.========")
            stats_1d["sequentials"] = generate_standard_plot_and_stats_for_1d_data(d, show_plot=False)
        else:
            print("=====statistics for qualitatives. Accuracy vs visualization.========")
            stats_1d["qualitatives"] = generate_standard_plot_and_stats_for_1d_data(d, show_plot=False)
    payload["stats_1d"] = stats_1d
    resp = get_main_stats(dfs, "colormap_type", ["sequential", "qualitative"], is_parametric=normality)
    payload.update(resp)
    return payload


def stats_means_sequential_qualitative_time(df: pd.DataFrame, dependent_variable_name: str = "time_metric") -> dict:
    heatmaps = df[df["visualization"] == "heatmap"]
    horanges = heatmaps[heatmaps["colormap"] == "oranges"]
    hpurple2 = heatmaps[heatmaps["colormap"] == "purple2"]
    hyellow_red = heatmaps[heatmaps["colormap"] == "yellow_red"]
    hset1 = heatmaps[heatmaps["colormap"] == "set1"]
    hmokole = heatmaps[heatmaps["colormap"] == "mokole"]
    hi_want_hue = heatmaps[heatmaps["colormap"] == "i_want_hue"]
    sequentials_df = pd.concat([horanges, hpurple2, hyellow_red])
    qualitatives_df = pd.concat([hset1, hmokole, hi_want_hue])

    def _calculate_metric(row: pd.Series):
        if (row["btw_start_and_first_change"] == row["btw_first_and_last_slider_action"]) and (
                row["btw_start_and_first_change"] != 0):
            return row["btw_start_and_first_change"]
        elif (row["btw_start_and_first_change"] == row["btw_first_and_last_slider_action"]) and (
                row["btw_start_and_first_change"] == 0):
            return row["time_delta_seconds"]
        else:
            return row["btw_start_and_first_change"] + row["btw_first_and_last_slider_action"]

    sequentials_df["time_metric"] = sequentials_df.apply(lambda row: _calculate_metric(row), axis=1)
    qualitatives_df["time_metric"] = qualitatives_df.apply(lambda row: _calculate_metric(row), axis=1)
    sequentials_df = sequentials_df[sequentials_df["time_metric"] > 0]
    qualitatives_df = qualitatives_df[qualitatives_df["time_metric"] > 0]
    sequentials = sequentials_df["time_metric"].to_list()
    qualitatives = qualitatives_df["time_metric"].to_list()
    dfs = [sequentials, qualitatives]
    payload, stats_1d = {}, {}
    for index, d in enumerate(dfs):
        if index == 0:
            print("=====statistics for  sequentials. Time till completion vs visualization.========")
            stats_1d["sequentials"] = generate_standard_plot_and_stats_for_1d_data(d, show_plot=False)
        else:
            print("=====statistics for qualitatives. Time till completion vs visualization.========")
            stats_1d["qualitatives"] = generate_standard_plot_and_stats_for_1d_data(d, show_plot=False)
    payload["stats_1d"] = stats_1d
    normality = True
    for l in dfs:
        if not is_data_normal(l):
            normality = False
    payload["are_populations_normal"] = normality
    resp = get_main_stats(dfs, "colormap_type", ["sequential", "qualitative"], is_parametric=normality)
    payload.update(resp)
    sequentials_df["pallette_type"] = "sequential"
    qualitatives_df["pallette_type"] = "qualitative"
    concated = pd.concat([sequentials_df, qualitatives_df], ignore_index=True, axis=0)
    concated.boxplot(column="time_metric", by=["pallette_type"])
    plt.title("pallette_type")
    plt.ylabel("time to solve task [s]")
    # plt.show()
    return payload


def stats_means_time_colormap(df: pd.DataFrame, dependent_variable_name: str = "time_metric") -> dict:
    line_chart_df = df[df["visualization"] == "line_chart"]
    heatmap_df = df[df["visualization"] == "heatmap"]

    def _calculate_metric(row: pd.Series):
        if (row["btw_start_and_first_change"] == row["btw_first_and_last_slider_action"]) and (
                row["btw_start_and_first_change"] != 0):
            return row["btw_start_and_first_change"]
        elif (row["btw_start_and_first_change"] == row["btw_first_and_last_slider_action"]) and (
                row["btw_start_and_first_change"] == 0):
            return row["time_delta_seconds"]
        else:
            return row["btw_start_and_first_change"] + row["btw_first_and_last_slider_action"]

    line_chart_df["time_metric"] = line_chart_df.apply(lambda row: _calculate_metric(row), axis=1)
    heatmap_df["time_metric"] = heatmap_df.apply(lambda row: _calculate_metric(row), axis=1)
    heatmap_df = heatmap_df[heatmap_df["time_metric"] > 0]
    line_chart_df = line_chart_df[line_chart_df["time_metric"] > 0]
    colormaps_lc = ["set1", "mokole", "i_want_hue"]
    colormaps_h = ["oranges", "purple2", "yellow_red", "set1", "mokole", "i_want_hue"]
    lc_dfs, h_dfs = [], []
    payload, stats_1d_lc, stats_1d_h = {}, {}, {}
    for index, lc_element in enumerate(colormaps_lc):
        instance_lc = line_chart_df[line_chart_df["colormap"] == colormaps_lc[index]][dependent_variable_name].to_list()
        lc_dfs.append(instance_lc)
        print(f"=====statistics for {colormaps_lc[index]}  line charts. Time till completion vs colormap.========")
        stats_1d_lc[colormaps_lc[index]] = generate_standard_plot_and_stats_for_1d_data(instance_lc, show_plot=False)
    for index, h_element in enumerate(colormaps_h):
        instance_h = heatmap_df[heatmap_df["colormap"] == colormaps_h[index]][dependent_variable_name].to_list()
        h_dfs.append(instance_h)
        print(f"=====statistics for {colormaps_h[index]}  colorfields. Time till completion vs colormap.========")
        stats_1d_h[colormaps_h[index]] = generate_standard_plot_and_stats_for_1d_data(instance_h, show_plot=False)
    payload["stats_1d_lc"] = stats_1d_lc
    payload["stats_1d_h"] = stats_1d_h
    lc_normality = True
    for l in lc_dfs:
        if not is_data_normal(l):
            lc_normality = False
    payload["are_lc_populations_normal"] = lc_normality
    h_normality = True
    for h in h_dfs:
        if not is_data_normal(h):
            h_normality = False
    resp = get_main_stats(lc_dfs, "lc", COLORMAPS_LINECHARTS, is_parametric=lc_normality)
    payload.update(resp)
    resp = get_main_stats(h_dfs, "h", COLORMAPS_COLORFIELDS, is_parametric=h_normality)
    payload.update(resp)
    line_chart_df.boxplot(column="time_metric", by=["colormap"])
    plt.title("line_chart")
    plt.ylabel("time to solve task [s]")
    # plt.show()
    heatmap_df["colormap"] = pd.Categorical(heatmap_df["colormap"], categories=["i_want_hue", "mokole", "set1", "oranges", "purple2", "yellow_red"])
    heatmap_df.sort_values("colormap", inplace=True)
    heatmap_df.boxplot(column="time_metric", by=["colormap"])
    plt.title("heatmap")
    plt.ylabel("time to solve task [s]")
    # plt.show()
    return payload


def stats_means_time_viz(df: pd.DataFrame) -> dict:
    line_charts_df = df[df["visualization"] == "line_chart"]
    heatmaps_df = df[df["visualization"] == "heatmap"]
    factor_levels = ["line_charts", "heatmaps"]
    def _calculate_metric(row: pd.Series):
        if (row["btw_start_and_first_change"] == row["btw_first_and_last_slider_action"]) and (
                row["btw_start_and_first_change"] != 0):
            return row["btw_start_and_first_change"]
        elif (row["btw_start_and_first_change"] == row["btw_first_and_last_slider_action"]) and (
                row["btw_start_and_first_change"] == 0):
            return row["time_delta_seconds"]
        else:
            return row["btw_start_and_first_change"] + row["btw_first_and_last_slider_action"]

    line_charts_df["time_metric"] = line_charts_df.apply(lambda row: _calculate_metric(row), axis=1)
    heatmaps_df["time_metric"] = heatmaps_df.apply(lambda row: _calculate_metric(row), axis=1)
    heatmaps_df = heatmaps_df[heatmaps_df["time_metric"] > 0]
    line_charts_df = line_charts_df[line_charts_df["time_metric"] > 0]
    line_charts = line_charts_df["time_metric"].to_list()
    heatmaps = heatmaps_df["time_metric"].to_list()
    dfs = [line_charts, heatmaps]
    payload, stats_1d = {}, {}
    for index, d in enumerate(dfs):
        if index == 0:
            print("=====statistics for  line charts. Time till completion vs visualization.========")
            stats_1d["line_charts"] = generate_standard_plot_and_stats_for_1d_data(d, show_plot=False)
        else:
            print("=====statistics for colorfields. Time till completion vs visualization.========")
            stats_1d["colorfields"] = generate_standard_plot_and_stats_for_1d_data(d, show_plot=False)
    payload["stats_1d"] = stats_1d
    normality = True
    for l in dfs:
        if not is_data_normal(l):
            normality = False
    payload["are_populations_normal"] = normality
    resp = get_main_stats(dfs, "viz", factor_levels, is_parametric=normality)
    payload.update(resp)
    concated = pd.concat([line_charts_df, heatmaps_df], ignore_index=True, axis=0)
    concated.boxplot(column="time_metric", by=["visualization"])
    plt.title("visualization")
    plt.ylabel("time to solve task [s]")
    # plt.show()
    return payload


def common_members_dfs(t1_df_merged, t2_df_merged, t3_df_merged, t4_df_merged, common_users):
    t0_common = t1_df_merged[t1_df_merged["userID"].isin(common_users)]
    t1_common = t2_df_merged[t2_df_merged["userID"].isin(common_users)]
    t2_common = t3_df_merged[t3_df_merged["userID"].isin(common_users)]
    t3_common = t4_df_merged[t4_df_merged["userID"].isin(common_users)]
    return t0_common, t1_common, t2_common, t3_common