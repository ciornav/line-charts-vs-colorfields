from scikit_posthocs import posthoc_dunn
from helpers.common_helpers import *
from helpers.study_helpers.moo_helpers import *
from study_analyzer import current_dir
from pymoo.problems import get_problem
import datetime
import re
import json
import copy


def task_1_analyzer(df: pd.DataFrame, user_config: pd.DataFrame) -> pd.DataFrame:
    # task 1 a empty state + initial state + each reset adds 2 trials
    eq_df = df[df["stepName"] == "task_0"]
    eq_df.reset_index(inplace=True)
    vals = eq_df["stepValue"].to_list()
    new_vals = []
    indexes_to_keep = []
    for index, value in enumerate(vals):
        val = remove_text_parsing_errors(value)
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
        number_of_outputs = int(len(row["final_slider_configuration"].keys()) / 2)
        try:
            out = "Model_" + str(number_of_outputs)
            row["electricity_cost"] = row["predictions"][out]
        except KeyError:
            row["electricity_cost"] = 1
        return row["electricity_cost"]

    eq_final.dropna(subset=["correctAnswer"], inplace=True)
    eq_final["electricity_cost"] = eq_final.apply(lambda row: add_predictions(row), axis=1)
    conv_1_4_task_number = 1
    history_costs = get_history_highest_and_lowest(eq_final, conv_1_4_task_number, user_config)
    history_df = pd.DataFrame(history_costs)
    eq_final = eq_final.merge(history_df, on="userID", how="inner")
    return eq_final


def task_2_analyzer(df: pd.DataFrame, user_config: pd.DataFrame) -> pd.DataFrame:
    # no empty state here. the initial state is recorded though. Reset takes 2 trials again.
    eq_df = df[df["stepName"] == "task_1"]
    eq_df.reset_index(inplace=True)
    vals = eq_df["stepValue"].to_list()
    new_vals = []
    indexes_to_keep = []
    for index, value in enumerate(vals):
        val = remove_text_parsing_errors(value)
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
    conv_1_4_task_number = 2
    history_costs = get_history_highest_and_lowest(eq_final, conv_1_4_task_number, user_config)
    history_df = pd.DataFrame(history_costs)
    eq_final = eq_final.merge(history_df, on="userID", how="inner")
    return eq_final


def task_3_analyzer(df: pd.DataFrame, user_config: pd.DataFrame) -> pd.DataFrame:
    # 2 trials for initial state. reset takes one trial.
    eq_df = df[df["stepName"] == "task_2"]
    eq_df.reset_index(inplace=True)
    vals = eq_df["stepValue"].to_list()
    new_vals = []
    indexes_to_keep = []
    for index, value in enumerate(vals):
        val = remove_text_parsing_errors(value)
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
    conv_1_4_task_number = 3
    history_costs = get_history_highest_and_lowest(eq_final, conv_1_4_task_number, user_config)
    history_df = pd.DataFrame(history_costs)
    eq_final = eq_final.merge(history_df, on="userID", how="inner")
    return eq_final


def task_4_analyzer(df: pd.DataFrame, user_config: pd.DataFrame) -> pd.DataFrame:
    eq_df = df[df["stepName"] == "task_3"]
    eq_df.reset_index(inplace=True)
    vals = eq_df["stepValue"].to_list()
    new_vals = []
    indexes_to_keep = []
    for index, value in enumerate(vals):
        val = remove_text_parsing_errors(value)
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
    conv_1_4_task_number = 4
    history_costs = get_history_highest_and_lowest(eq_final, conv_1_4_task_number, user_config)
    history_df = pd.DataFrame(history_costs)
    eq_final = eq_final.merge(history_df, on="userID", how="inner")
    return eq_final


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


def stats_means_accuracy(df: pd.DataFrame, dependent_variable_name: str) -> dict:
    line_chart_df = df[df["visualization"] == "line_chart"]
    # line_chart_df = df[df["visualization"] == "line_chart"][dependent_variable_name].to_list()
    heatmap_df = df[df["visualization"] == "heatmap"]
    # heatmap_df = df[df["visualization"] == "heatmap"][dependent_variable_name].to_list()
    datasets = list(line_chart_df["dataset"].unique())
    outputs_config = list(line_chart_df["number_of_outputs"].unique())
    fig, axs = plt.subplots(nrows=len(datasets) + 1, ncols=len(outputs_config) + 1, figsize=(15.5,11))
    fig.suptitle(f"Results for {df.columns[2]}")
    conv_1_4_task_number = int(str(df.columns[2])[-1])
    results_dataset_config = {}
    scaled_results = {}
    for row, dataset in enumerate(datasets):
        for column, output_config in enumerate(outputs_config):
            lc_filt1 = line_chart_df[line_chart_df["dataset"] == dataset]
            lc_list = lc_filt1[lc_filt1["number_of_outputs"] == output_config][dependent_variable_name].to_list()
            h_filt1 = heatmap_df[heatmap_df["dataset"] == dataset]
            h_list = h_filt1[h_filt1["number_of_outputs"] == output_config][dependent_variable_name].to_list()
            config = dependent_variable_name + "_" + dataset + "_" + str(output_config)
            starting_point = get_normalized_starting_point(dataset, int(output_config), conv_1_4_task_number)
            best_point = get_best_solution(dataset, int(output_config), conv_1_4_task_number)
            scaled_lc = rescale_results(best=best_point, starting=starting_point, value=lc_list,
                                        task_number=conv_1_4_task_number)
            scaled_h = rescale_results(best=best_point, starting=starting_point, value=h_list, task_number=conv_1_4_task_number)
            scaled_results[config] = {"line_charts": scaled_lc, "colorfields": scaled_h}
            results_dataset_config[config] = check_normality_and_apply_statistical_test([scaled_lc, scaled_h])
            axs[row, column].boxplot([scaled_lc, scaled_h], showfliers=True, showmeans=True)
            axs[row, column].set_title(config)
            axs[row, column].set_ylabel(dependent_variable_name)
            axs[row, column].set_xticklabels(["line charts", "colorfields"])
            axs[row, column].set_ylim(-0.1, 1.1)
            axs[row, column].grid(True)
    for row, dataset in enumerate(datasets):
        config = dependent_variable_name + "_dataset_" + dataset
        filtered_data_labels = [key for key in scaled_results.keys() if dataset in key]
        lcs = [scaled_results[item]["line_charts"] for item in filtered_data_labels]
        hs = [scaled_results[item]["colorfields"] for item in filtered_data_labels]
        lc_filt1 = [item for sublist in lcs for item in sublist]
        h_filt1 = [item for sublist in hs for item in sublist]
        axs[row, len(outputs_config)].boxplot([lc_filt1, h_filt1], showfliers=True, showmeans=True)
        axs[row, len(outputs_config)].set_title(config)
        axs[row, len(outputs_config)].set_ylabel(dependent_variable_name)
        axs[row, len(outputs_config)].set_xticklabels(["line charts", "colorfields"])
        results_dataset_config[config] = check_normality_and_apply_statistical_test([lc_filt1, h_filt1])
        axs[row, len(outputs_config)].set_ylim(-0.1, 1.1)
        axs[row, len(outputs_config)].grid(True)
    for row, output_config in enumerate(outputs_config):
        config = dependent_variable_name + "_number_of_outputs_" + output_config
        filtered_data_labels = [key for key in scaled_results.keys() if ("_" + str(output_config)) in key]
        lcs = [scaled_results[item]["line_charts"] for item in filtered_data_labels]
        hs = [scaled_results[item]["colorfields"] for item in filtered_data_labels]
        lc_filt1 = [item for sublist in lcs for item in sublist]
        h_filt1 = [item for sublist in hs for item in sublist]
        axs[len(datasets), row].boxplot([lc_filt1, h_filt1], showfliers=True, showmeans=True)
        axs[len(datasets), row].set_title(config)
        axs[len(datasets), row].set_ylabel(dependent_variable_name)
        axs[len(datasets), row].set_xticklabels(["line charts", "colorfields"])
        results_dataset_config[config] = check_normality_and_apply_statistical_test([lc_filt1, h_filt1])
        axs[len(datasets), row].set_ylim(-0.1, 1.1)
        axs[len(datasets), row].grid(True)
    # plt.subplots_adjust(top=1)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    lcs = [item["line_charts"] for item in scaled_results.values()]
    hs = [item["colorfields"] for item in scaled_results.values()]
    lc_filt1 = [item for sublist in lcs for item in sublist]
    h_filt1 = [item for sublist in hs for item in sublist]
    axs[len(datasets), len(outputs_config)].boxplot([lc_filt1, h_filt1], showfliers=True, showmeans=True)
    axs[len(datasets),len(outputs_config)].set_title(f"task {conv_1_4_task_number}: line charts vs. colorfields")
    axs[len(datasets), len(outputs_config)].set_ylabel(dependent_variable_name)
    axs[len(datasets), len(outputs_config)].set_xticklabels(["line charts", "colorfields"])
    axs[len(datasets), len(outputs_config)].grid(True)
    results_dataset_config[df.columns[2]] = check_normality_and_apply_statistical_test(
        [lc_filt1, h_filt1])
    fig.tight_layout(rect=[0, 0.03,  1, 0.95])
    return results_dataset_config


def stats_time(df: pd.DataFrame) -> dict:
    line_chart_df = df[df["visualization"] == "line_chart"]
    heatmap_df = df[df["visualization"] == "heatmap"]
    dependent_variable_name = "time_metric"
    def _calculate_metric(row: pd.Series):
        if (row["btw_start_and_first_change"] == row["btw_first_and_last_slider_action"]) and (
                row["btw_start_and_first_change"] != 0):
            return row["btw_start_and_first_change"]
        elif (row["btw_start_and_first_change"] == row["btw_first_and_last_slider_action"]) and (
                row["btw_start_and_first_change"] == 0):
            return row["time_delta_seconds"]
        else:
            return row["btw_start_and_first_change"] + row["btw_first_and_last_slider_action"]

    line_chart_df[dependent_variable_name] = line_chart_df.apply(lambda row: _calculate_metric(row), axis=1)
    heatmap_df[dependent_variable_name] = heatmap_df.apply(lambda row: _calculate_metric(row), axis=1)
    # heatmap_df = df[df["visualization"] == "heatmap"][dependent_variable_name].to_list()
    datasets = list(line_chart_df["dataset"].unique())
    outputs_config = list(line_chart_df["number_of_outputs"].unique())
    fig, axs = plt.subplots(nrows=len(datasets) + 1, ncols=len(outputs_config) + 1, figsize=(16,10))
    fig.suptitle(f"Results for {df.columns[2]}")
    conv_1_4_task_number = int(str(df.columns[2])[-1])
    results_dataset_config = {}
    scaled_results = {}
    for row, dataset in enumerate(datasets):
        for column, output_config in enumerate(outputs_config):
            lc_filt1 = line_chart_df[line_chart_df["dataset"] == dataset]
            lc_list = lc_filt1[lc_filt1["number_of_outputs"] == output_config][dependent_variable_name].to_list()
            h_filt1 = heatmap_df[heatmap_df["dataset"] == dataset]
            h_list = h_filt1[h_filt1["number_of_outputs"] == output_config][dependent_variable_name].to_list()
            config = dependent_variable_name + "_" + dataset + "_" + str(output_config)
            scaled_results[config] = {"line_charts": lc_list, "colorfields": h_list}
            results_dataset_config[config] = check_normality_and_apply_statistical_test([lc_list, h_list])
            axs[row, column].boxplot([lc_list, h_list], showfliers=True, showmeans=True)
            axs[row, column].set_title(config)
            axs[row, column].set_ylabel("[s]")
            axs[row, column].grid(True)
            axs[row, column].set_xticklabels(["line charts", "colorfields"])
    for row, dataset in enumerate(datasets):
        config = dependent_variable_name + "_dataset_" + dataset
        filtered_data_labels = [key for key in scaled_results.keys() if dataset in key]
        lcs = [scaled_results[item]["line_charts"] for item in filtered_data_labels]
        hs = [scaled_results[item]["colorfields"] for item in filtered_data_labels]
        lc_filt1 = [item for sublist in lcs for item in sublist]
        h_filt1 = [item for sublist in hs for item in sublist]
        axs[row, len(outputs_config)].boxplot([lc_filt1, h_filt1], showfliers=True, showmeans=True)
        axs[row, len(outputs_config)].set_title(config)
        axs[row, len(outputs_config)].set_ylabel("[s]")
        axs[row, len(outputs_config)].grid(True)
        axs[row, len(outputs_config)].set_xticklabels(["line charts", "colorfields"])
        results_dataset_config[config] = check_normality_and_apply_statistical_test([lc_filt1, h_filt1])
    for row, output_config in enumerate(outputs_config):
        config = dependent_variable_name + "_number_of_outputs_" + output_config
        filtered_data_labels = [key for key in scaled_results.keys() if ("_" + str(output_config)) in key]
        lcs = [scaled_results[item]["line_charts"] for item in filtered_data_labels]
        hs = [scaled_results[item]["colorfields"] for item in filtered_data_labels]
        lc_filt1 = [item for sublist in lcs for item in sublist]
        h_filt1 = [item for sublist in hs for item in sublist]
        axs[len(datasets), row].boxplot([lc_filt1, h_filt1], showfliers=True, showmeans=True)
        axs[len(datasets), row].set_title(config)
        axs[len(datasets), row].set_ylabel("[s]")
        axs[len(datasets), row].grid(True)
        axs[len(datasets), row].set_xticklabels(["line charts", "colorfields"])
        results_dataset_config[config] = check_normality_and_apply_statistical_test([lc_filt1, h_filt1])
    # plt.subplots_adjust(top=1)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    lcs = [item["line_charts"] for item in scaled_results.values()]
    hs = [item["colorfields"] for item in scaled_results.values()]
    lc_filt1 = [item for sublist in lcs for item in sublist]
    h_filt1 = [item for sublist in hs for item in sublist]
    axs[len(datasets), len(outputs_config)].boxplot([lc_filt1, h_filt1], showfliers=True, showmeans=True)
    axs[len(datasets),len(outputs_config)].set_title(f"task {conv_1_4_task_number}: line charts vs. colorfields")
    axs[len(datasets), len(outputs_config)].set_ylabel('[s]')
    axs[len(datasets), len(outputs_config)].set_xticklabels(["line charts", "colorfields"])
    axs[len(datasets), len(outputs_config)].grid(True)
    results_dataset_config[df.columns[2]] = check_normality_and_apply_statistical_test(
        [lc_filt1, h_filt1])
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    return results_dataset_config


def stats_means_time_viz(df: pd.DataFrame) -> dict:
    line_charts_df = df[df["visualization"] == "line_chart"]
    heatmaps_df = df[df["visualization"] == "heatmap"]
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
    lc_normality = True
    for l in dfs:
        if not is_data_normal(l):
            lc_normality = False
    payload["are_populations_normal"] = lc_normality
    if lc_normality:
        are_lc_means_different = is_the_mean_different(dfs)
        res_lc = tukey_hsd(*dfs)
        payload["tukey_hsd_lc"] = res_lc
    else:
        are_lc_means_different = is_the_mean_different(dfs, method="kruskal")
        res_lc = posthoc_dunn(dfs, p_adjust='holm')
        payload["dunn_lc"] = res_lc
    payload["are_means_different"] = are_lc_means_different
    concated = pd.concat([line_charts_df, heatmaps_df], ignore_index=True, axis=0)
    concated.boxplot(column="time_metric", by=["visualization"])
    plt.title("visualization")
    plt.ylabel("time to solve task [s]")
    # plt.show()
    return payload


def rescale_results(best: float, starting: float, value: [float], task_number: int) -> [float]:
    l = []
    # for val in value:
    #     l.append((val - best) / (starting - best))
    if task_number < 4:
        for val in value:
            l.append((val - best) / (starting - best))
    else:
        for val in value:
            l.append((val - starting) / (best - starting))
            # l.append(1 - (val - starting) / (best - starting))
    if sum(np.array(l) > 1):
        print("rescaled higher than 1")
    multi_move_issue_fix = [1 if item > 1 else item for item in l]
    return multi_move_issue_fix


def get_normalized_starting_point(dataset: str, output_config: int, conv1_4_task: int) -> float:
    data = get_nadir_data(dataset, output_config)
    top20 = data["top20"]
    nadir_closest = top20[0]["y"]
    if conv1_4_task != 4:
        if conv1_4_task == 1:
            elec_cost = nadir_closest[-1]
            normalized_res = elec_cost / (data["highest_ys"][-1] * 1.2)
        else:
            normalized_res = normalize_problem_predictions_and_get_average(data)
        return normalized_res
    else:
        file_name = dataset.lower() + str(output_config * 2) + str(output_config) + "_less_stable_points.csv"
        file_path = current_dir / "data/study_data/nadir" / file_name
        df = pd.read_csv(file_path)
        df.sort_values(by="differences", ascending=False, inplace=True)
        df.reset_index(drop=True, inplace=True)
        problem = get_problem(name=dataset.lower(), n_var=output_config * 2, n_obj=output_config)
        x = np.array(eval(df["approximated_pareto_points"][0]))
        res = problem.evaluate(x)
        norms = []
        for index, val in enumerate(res):
            norms.append(val / (data["highest_ys"][index] * 1.2))
        return np.sum(norms)


def get_best_solution(dataset: str, output_config: int, task_number: int) -> float:
    results = []
    conv_1_4_task = task_number
    if conv_1_4_task == 4:
        file_name = dataset.lower() + str(output_config * 2) + str(output_config) + "_less_stable_points.csv"
        file_path = current_dir / "data/study_data/nadir" / file_name
        df = pd.read_csv(file_path)
        df.sort_values(by="differences", ascending=False, inplace=True)
        df.reset_index(inplace=True, drop=True)
        problem = get_problem(name=dataset.lower(), n_var=output_config * 2, n_obj=output_config)
        x = np.array(eval(df["grid_points_with_highest_costs"][0]))
        res = problem.evaluate(x)
        nadir_data = get_nadir_data(dataset, output_config)
        for index, val in enumerate(res):
            results.append(val / (nadir_data["highest_ys"][index] * 1.2))
        best_result = np.sum(results)
    elif conv_1_4_task == 3:
        file_name = "pareto_" + dataset.lower() + "_results.json"
        file_path = current_dir / "data/study_data/nadir" / file_name
        with open(file_path, 'r', encoding='utf-8') as f:
            pareto_res = json.load(f)
        pareto = pareto_res[dataset.lower() + "_" + str(output_config)]
        nadir_data = np.array(get_nadir_data(dataset, output_config)["highest_ys"])
        normalized_pareto = [(point/(nadir_data*1.2)) for point in pareto]
        agg_pareto = [np.sum(i) for i in normalized_pareto]
        best_result = min(agg_pareto)
    elif conv_1_4_task == 1:
        problem = get_problem(name=dataset.lower(), n_var=output_config * 2, n_obj=output_config)
        sliders_grid = get_slider_grid(problem)
        nadir_data = get_nadir_data(dataset, output_config)
        top20 = nadir_data["top20"]
        nadir_closest = top20[0]["x"]
        result_nadir_closest = problem.evaluate(np.array(nadir_closest))
        best_result = result_nadir_closest[-1] / (nadir_data["highest_ys"][-1] * 1.2)
        for key, value in sliders_grid.items():
            slightly_modified_nadir = copy.deepcopy(nadir_closest)
            position = int(key.split("_")[-1])
            for index, tick in enumerate(value):
                slightly_modified_nadir[position] = tick
                temp_res = problem.evaluate(np.array(slightly_modified_nadir))
                normalized_temp_res = temp_res[-1] / (nadir_data["highest_ys"][-1] * 1.2)
                if normalized_temp_res < best_result:
                    best_result = normalized_temp_res
    else:  # i.e. task 2
        problem = get_problem(name=dataset.lower(), n_var=output_config * 2, n_obj=output_config)
        sliders_grid = get_slider_grid(problem)
        nadir_data = get_nadir_data(dataset, output_config)
        top20 = nadir_data["top20"]
        nadir_closest = top20[0]["x"]
        result_nadir_closest = problem.evaluate(np.array(nadir_closest))
        best_result = np.sum([v / (nadir_data["highest_ys"][i] * 1.2) for i, v in enumerate(result_nadir_closest)])
        for key, value in sliders_grid.items():
            slightly_modified_nadir = copy.deepcopy(nadir_closest)
            position = int(key.split("_")[-1])
            for index, tick in enumerate(value):
                slightly_modified_nadir[position] = tick
                temp_res = problem.evaluate(np.array(slightly_modified_nadir))
                normalized_temp_res = [v / (nadir_data["highest_ys"][i] * 1.2) for i, v in enumerate(temp_res)]
                if np.sum(normalized_temp_res) < best_result:
                    best_result = np.sum(normalized_temp_res)
    return best_result


def get_nadir_data(dataset: str, output_config: int) -> [float]:
    file_name = dataset.lower() + str(output_config * 2) + str(output_config) + "_top_nadir.json"
    file_path = current_dir / "data/study_data/nadir" / file_name
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data


def normalize_problem_predictions_and_get_average(data: {}):
    normalized_predictions = []
    top20 = data["top20"]
    nadir_closest = top20[0]["y"]
    for index, pred in enumerate(nadir_closest):
        maximum = data["highest_ys"][index]*1.2
        normalized_prediction = pred / maximum
        normalized_predictions.append(normalized_prediction)
    return np.sum(normalized_predictions)


def check_normality_and_apply_statistical_test(list_of_groups: [], name: str = "standard") -> {}:
    dfs = list_of_groups
    payload = {}
    normality = True
    for d in dfs:
        if not is_data_normal(d):
            normality = False
    payload["are_populations_normal"] = normality
    factor_levels = ["line_charts", "colorfields"]
    if normality:
        are_means_different = is_the_mean_different(dfs)
        payload["effect_size"] = get_global_effect_size(dfs, method="one_way_anova")
        res_lc = tukey_hsd(*dfs)
        payload["tukey_hsd"] = res_lc
        pairwise_effects = get_parametric_pairwise_effect_size(dfs, factor_levels)
        payload["pairwise_effects"] = {"pairwise_effects": pairwise_effects, "type": "parametric"}
        ci = get_confidence_intervals_parametric(dfs, factor_levels)
        payload["confidence_intervals"] = ci
    else:
        are_means_different = is_the_mean_different(dfs, method="kruskal")
        payload["effect_size"] = get_global_effect_size(dfs, method="kruskal")
        res_lc = posthoc_dunn(dfs, p_adjust='holm')
        payload["dunn"] = res_lc
        pairwise_effects = get_non_parametric_pairwise_effect_size(dfs, factor_levels)
        payload["pairwise_effects"] = {"pairwise_effects": pairwise_effects, "type": "non_parametric"}
        confidence_intervals = get_confidence_intervals_non_parametric(dfs, factor_levels)
        payload["confidence_intervals"] = confidence_intervals
    payload["are_means_different"] = are_means_different
    return {name: payload}


def stats_means_accuracy_viz(df: pd.DataFrame, dependent_variable_name: str) -> dict:
    line_charts = df[df["visualization"] == "line_chart"][dependent_variable_name].to_list()
    heatmaps = df[df["visualization"] == "heatmap"][dependent_variable_name].to_list()
    dfs = [line_charts, heatmaps]
    payload = {}
    lc_normality = True
    for l in dfs:
        if not is_data_normal(l):
            lc_normality = False
    payload["are_populations_normal"] = lc_normality
    stats_1d = {}
    for index, d in enumerate(dfs):
        if index == 0:
            print("=====statistics for  line charts. Accuracy vs visualization.========")
            stats_1d["line_charts"] = generate_standard_plot_and_stats_for_1d_data(d, show_plot=False)
        else:
            print("=====statistics for colorfields. Accuracy vs visualization.========")
            stats_1d["colorfields"] = generate_standard_plot_and_stats_for_1d_data(d, show_plot=False)
    payload["stats_1d"] = stats_1d
    if lc_normality:
        are_lc_means_different = is_the_mean_different(dfs)
        res_lc = tukey_hsd(*dfs)
        payload["tukey_hsd_lc"] = res_lc
    else:
        are_lc_means_different = is_the_mean_different(dfs, method="kruskal")
        res_lc = posthoc_dunn(dfs, p_adjust='holm')
        payload["dunn_lc"] = res_lc
    payload["are_means_different"] = are_lc_means_different
    return payload


def clean_raw_df(df: pd.DataFrame) -> pd.DataFrame:
    # remove special characters from participants answers such as // or / or other
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
            row[2] = re.sub(
                r'repeat until there-s no more adjusting needed. Pay attention to the .*total cost.* bar in the Cost Overview Panel to see if',
                'repeat until there-s no more adjusting needed. Pay attention to the total cost bar in the Cost Overview Panel to see if',
                row[2])
            row[2] = re.sub(r'I was confused because I saw the number .*2.* as the place where I moved the slider',
                            'I was confused because I saw the number 2 as the place where I moved the slider', row[2])
            row[2] = re.sub(r'whereas other options seemed to .*balance out.* any differences more',
                            'whereas other options seemed to balance out any differences more', row[2])
            row[2] = re.sub(r'Find the .*X.* axis with highest volume threshold',
                            'Find the X axis with highest volume threshold', row[2])
            row[2] = re.sub(r'and most were se to the  .*max.* of the costs, so moving it',
                            'and most were se to the  max of the costs, so moving it', row[2])
            row[2] = re.sub(r' Hard to tell if .*three steps.* meant the integer',
                            ' Hard to tell if three steps meant the integer', row[2])
            row[2] = re.sub(r'Setting_1 and Setting_2 had the .*steepest gradients.* of color change',
                            'Setting_1 and Setting_2 had the steepest gradients of color change', row[2])
            row[2] = re.sub(r'I then moved both sliders to confirm if this would remain .*true.* after x-axis changes',
                            'I then moved both sliders to confirm if this would remain valid after x-axis changes',
                            row[2])
            return row
        except AttributeError:
            return row

    df.apply(lambda row: clean_row(row), axis=1)
    return df


def get_list_of_invalid_submissions() -> dict:
    # a list of submissions with invalid data. Invalid due to either aborting or very early terminations or user comments
    users_and_tasks_to_delete = {
        "D7636161CBD246C8A495F4FCD7CFD672": [],
        "67CF9A0E1DE7432D878DC83B71F94130": [],
        "8B7F729B4B04419CB3367CF9BB0CC821": [],
        "5578D512273A4538B9040D47DFFBA554": [],
        "F535B74B17A14F078DD3CEB5A281D71B": [],
        "1A53FABA0FF043188F998EDB91BBA2F6": [],
        "B9E8098AFB14466EB2D448FD1F79FAB0": [],
        "Vasile1": [],
        "Vasile": [],
        # "34F3FBEBA1E34CAC9AFA473734919728": [],
        # "0A8F98AE54CE4B5BB8D594C210809C82": [],
    }
    return users_and_tasks_to_delete


def remove_invalid_submissions(df: pd.DataFrame):
    df = df[1:]
    users_to_delete = get_list_of_invalid_submissions()
    df.reset_index(inplace=True, drop=True)
    indexes_to_remove = []
    for index, val in df["userID"].items():
        if val in users_to_delete.keys():
            indexes_to_remove.append(index)
    good_labels = [i for i in list(df.index) if df.loc[i, "userID"] not in list(users_to_delete.keys())]
    after_user_removal_df = df.iloc[good_labels, :].reset_index()
    return after_user_removal_df


def remove_first_and_last_lines(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()

    # Remove the first line and the last three lines
    processed_lines = lines[1:-2]
    processed_file_name = filename.replace(".csv", "_processed.csv")
    with open(processed_file_name, 'w') as file:
        file.writelines(processed_lines)


def remove_text_parsing_errors(value: str) -> str:
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


def get_history_highest_and_lowest(df: pd.DataFrame, conv_1_4_task_number: int,
                                   user_config: pd.DataFrame) -> dict:
    #  there is a shift btw slider configs and predictions. Predictions are for the previous config of the slider. Late 1 step.
    history_lowest_overall_cost = []
    history_highest_overall_cost = []
    history_highest_elec_cost = []
    history_lowest_elec_cost = []
    timestamps = []
    userIds = []
    for index, el in enumerate(df["history"]):
        user_id = df.iloc[index]["userID"]
        try:
            user_configuration = user_config.loc[
                user_config["userID"] == user_id, ["number_of_inputs", "number_of_outputs", "dataset"]].to_dict(
                orient="records")[0]
        except IndexError:
            print(
                f"user {user_id} has not reached the step where it gets a configuration and aborted the study.")
            continue
        userIds.append(user_id)
        user_timestamps = {}
        individual_timestamps = []
        temp_list = []
        elec_list = []
        starting_point = {}
        is_starting_point_reached = False
        for ind, item in enumerate(el):
            if not is_starting_point_reached:
                is_starting_point_reached = isStartingPointReached(item["sliderConfig"], conv_1_4_task_number,
                                                                   user_configuration)
                # if task_number == 2:
                #     is_starting_point_reached = True
                starting_point = item["sliderConfig"]
                if not is_starting_point_reached:
                    continue
            changes = check_number_of_changes_compared_to_starting_point(starting_point, item["sliderConfig"])
            if conv_1_4_task_number in [1, 2, 4] and changes > 1:
                continue
            if len(item["predictions"].values()) == 0:
                continue
            if item["sliderConfig"] == starting_point:
                continue  # removes the resets
            temp_list.append(sum(item["predictions"].values()))
            number_of_outputs = int(len(starting_point.keys()) / 2)
            out = "Model_" + str(number_of_outputs)
            elec_list.append(item["predictions"][out])
            individual_timestamps.append(item["historyTime"])
        user_timestamps[df.iloc[index]["userID"]] = individual_timestamps
        timestamps.append(user_timestamps)
        dataset = user_configuration["dataset"]
        output_config = int(user_configuration["number_of_outputs"])
        if len(elec_list) > 0:
            if conv_1_4_task_number == 1:
                history_highest_elec_cost.append(
                    max(correct_list_if_reset(dataset, output_config, conv_1_4_task_number, elec_list)))
                history_lowest_elec_cost.append(
                    min(correct_list_if_reset(dataset, output_config, conv_1_4_task_number, elec_list)))
            else:
                history_highest_elec_cost.append(
                    max(elec_list))
                history_lowest_elec_cost.append(
                    min(elec_list))
        else:
            history_highest_elec_cost.append("nan")
            history_lowest_elec_cost.append("nan")
        if len(temp_list) > 0:
            if conv_1_4_task_number == 1:
                history_lowest_overall_cost.append(min(temp_list))
                history_highest_overall_cost.append(max(temp_list))
            else:
                history_lowest_overall_cost.append(
                    min(correct_list_if_reset(dataset, output_config, conv_1_4_task_number, temp_list)))
                history_highest_overall_cost.append(
                    max(correct_list_if_reset(dataset, output_config, conv_1_4_task_number, temp_list)))
        else:
            history_lowest_overall_cost.append("nan")
            history_highest_overall_cost.append("nan")
    return {
        "userID": userIds,
        "history_lowest_overall_cost": history_lowest_overall_cost,
        "history_highest_overall_cost": history_highest_overall_cost,
        "history_highest_elec_cost": history_highest_elec_cost,
        "history_lowest_elec_cost": history_lowest_elec_cost,
    }


def correct_list_if_reset(dataset: str, output_config: int, conv_1_4_task_number: int,
                          list_of_results: [float]):
    starting_point = get_normalized_starting_point(dataset, int(output_config), conv_1_4_task_number)
    best_point = get_best_solution(dataset, int(output_config), conv_1_4_task_number)
    if starting_point > best_point:
        # Removing invalid points. In some rare cases due to the automatic slider reset (users not respecting the ask),
        # the models are updates sequentially and might end up with better values (compared to best point) temporarily.
        # Filtering these results for task 1,2.
        new_list = [item for item in list_of_results if
                    (item >= best_point * 0.99)]  # 0.99 is just for rounding issues
    else:
        # Filtering these results for task 4. Direction is opposite.
        new_list = [item for item in list_of_results if (item <= best_point)]
    return new_list


def isStartingPointReached(payload: dict, conv_1_4_task_number: int, user_configuration: dict) -> bool:
    starting_point = {}
    if len(payload.keys()) == 0:
        return False
    if conv_1_4_task_number in [1, 2, 3]:
        file_name = user_configuration["dataset"].lower() + user_configuration["number_of_inputs"] + \
                    user_configuration[
                        "number_of_outputs"] + "_top_nadir.json"
        file_path = current_dir / "data/study_data/nadir" / file_name
        with open(file_path, 'r') as file:
            data = json.load(file)
        top20 = data["top20"]
        defaults_on_x = top20[0]["x"]
        starting_point = dict(zip(payload.keys(), defaults_on_x))
    if conv_1_4_task_number == 4:
        file_name = user_configuration["dataset"].lower() + user_configuration["number_of_inputs"] + \
                    user_configuration[
                        "number_of_outputs"] + "_less_stable_points.csv"
        file_path = current_dir / "data/study_data/nadir" / file_name
        df = pd.read_csv(file_path)
        df.sort_values(by="differences", ascending=False, inplace=True)
        df.reset_index(inplace=True, drop=True)
        defaults_on_x = eval(df["approximated_pareto_points"][0])
        starting_point = dict(zip(payload.keys(), defaults_on_x))
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
        time_delta = datetime.datetime.strptime(task_end_time,
                                                "%Y-%m-%dT%H:%M:%S.%fZ") - datetime.datetime.strptime(
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


def data_set_creator(df: pd.DataFrame):
    cvd_df = cvd_analyzer(df)
    eq_df = entry_questionnaire_analyzer(df)
    end_q_df = end_questionnaire_analyzer(df)
    user_config_df = eq_df.loc[:,
                     ["userID", "visualization", "colormap", "scent_height", "number_of_inputs", "number_of_outputs",
                      "dataset"]]
    t0_df = task_1_analyzer(df, user_config_df)
    t0_df_merged = t0_df.merge(user_config_df, on="userID", how="inner")
    t1_df = task_2_analyzer(df, user_config_df)
    t1_df_merged = t1_df.merge(user_config_df, on="userID", how="inner")
    t2_df = task_3_analyzer(df, user_config_df)
    t2_df_merged = t2_df.merge(user_config_df, on="userID", how="inner")
    t3_df = task_4_analyzer(df, user_config_df)
    t3_df_merged = t3_df.merge(user_config_df, on="userID", how="inner")
    return eq_df, cvd_df, t0_df_merged, t1_df_merged, t2_df_merged, t3_df_merged, end_q_df


def data_set_cleaner(df: pd.DataFrame) -> pd.DataFrame:
    indexes_to_remove = []
    task_number = [i for i in list(df.columns) if i.__contains__("task")][0]
    print(f"cleaning for task number: {task_number}")
    for index, row in df.iterrows():
        if row["btw_first_and_last_slider_action"] < 0.1:
            indexes_to_remove.append(index)
            removed_user = df.loc[index, "userID"]
            print(f"user {removed_user} was removed due to now time btw first and last slider move")
            continue
        if "task3" in list(row.index):
            if row["time_delta_seconds"] > 300 or row["time_delta_seconds"] < 10:
                indexes_to_remove.append(index)
                removed_user = df.loc[index, "userID"]
                print(
                    f"user {removed_user} was removed due to now longer time spent on task. Beyond 200 or 300 seconds")
                continue
        else:
            if row["time_delta_seconds"] > 200 or row["time_delta_seconds"] < 10:
                indexes_to_remove.append(index)
                removed_user = df.loc[index, "userID"]
                print(
                    f"user {removed_user} was removed due to now longer time spent on task. Beyond 200 or 300 seconds")
                continue
        if row["number_of_trials"] < 1:
            indexes_to_remove.append(index)
            removed_user = df.loc[index, "userID"]
            print(f"user {removed_user} was removed due to no trials on task.")
            continue
        if row["median_viz_loading_time_per_slider_ms"] > 15000:
            indexes_to_remove.append(index)
            removed_user = df.loc[index, "userID"]
            print(f"user {removed_user} was removed due to long visualization loading time.")
    return df.drop(indexes_to_remove)


def get_1d_stats_on_experiment(line_chart_df: pd.DataFrame, heatmap_df: pd.DataFrame, dependent_variable_name: str, task_number: str):
    lc_stats = []
    for dimensionality in line_chart_df["number_of_outputs"].unique():
        metric_subset = line_chart_df[line_chart_df["number_of_outputs"] == dimensionality][
            dependent_variable_name].reset_index(drop=True)
        print(
            f"===============generating statistics for task {task_number} for line charts and dimensionality: {dimensionality} ==================")
        d = generate_standard_plot_and_stats_for_1d_data(metric_subset, show_plot=False)
        d["name"] = dimensionality
        lc_stats.append(d)
    h_stats = []
    for dimensionality in heatmap_df["number_of_outputs"].unique():
        metric_subset = heatmap_df[heatmap_df["number_of_outputs"] == dimensionality][
            dependent_variable_name].reset_index(drop=True)
        print(
            f"===============generating statistics for task {task_number} for colorfields and dinensionality: {dimensionality} ==================")
        d = generate_standard_plot_and_stats_for_1d_data(metric_subset, show_plot=False)
        d["name"] = dimensionality
        h_stats.append(d)
    return lc_stats, h_stats