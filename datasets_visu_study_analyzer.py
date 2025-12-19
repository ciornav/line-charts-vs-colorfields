import pathlib
from typing import Dict
from helpers.study_helpers.general_study_helpers import *

current_dir = pathlib.Path(__file__).parent.absolute()


def read_data():
    filename = "./data/study_data/participant_data.csv"
    remove_first_and_last_lines(filename)
    processed_file_name = filename.replace(".csv", "_processed.csv")
    df = pd.read_csv(processed_file_name, delimiter="|")
    df = clean_raw_df(df)
    df.columns = ["userID", "stepName", "stepValue", "studyID", "sessionID", "timeBackend"]
    df = remove_invalid_submissions(df)
    return df


def task_plotter(task_data: pd.DataFrame, task_number: int, dependent_variable_name: str = "") -> dict:
    datasets = {
        "WFG2_lc": task_data[(task_data["dataset"] == "WFG2") & (task_data["visualization"] == "line_chart")],
        "WFG4_lc": task_data[(task_data["dataset"] == "WFG4") & (task_data["visualization"] == "line_chart")],
        "WFG5_lc": task_data[(task_data["dataset"] == "WFG5") & (task_data["visualization"] == "line_chart")],
        "WFG2_h": task_data[(task_data["dataset"] == "WFG2") & (task_data["visualization"] == "heatmap")],
        "WFG4_h": task_data[(task_data["dataset"] == "WFG4") & (task_data["visualization"] == "heatmap")],
        "WFG5_h": task_data[(task_data["dataset"] == "WFG5") & (task_data["visualization"] == "heatmap")],
    }
    dataset_stats = get_1d_stats_on_exp(datasets, dependent_variable_name, task_number)
    # these are all the statistical tests
    # the statistics for each of the 1d data are stored in lc_stats for line_charts and in h_stats for heatmaps
    means_stats_payload = stats_avg_accuracy(datasets, dependent_variable_name, f"task_{task_number}")
    time_stats_payload = get_time_statistics(datasets, task_number=f"task_{task_number}")
    full_stats = {"normalized_costs_task1": means_stats_payload,
                  "time_stats_task1": time_stats_payload}
    return full_stats


def get_1d_stats_on_exp(datasets: Dict[str, pd.DataFrame], dependent_variable_name: str, task_number: str) -> Dict[str, list]:
    datasets_stats = {}
    for dataset_name, dataset_df in datasets.items():
        dataset_stats = []
        for dimensionality in dataset_df["number_of_outputs"].unique():
            metric_subset = dataset_df[dataset_df["number_of_outputs"] == dimensionality][
            dependent_variable_name].reset_index(drop=True)
            print(
            f"===============generating statistics for task {task_number} for {dataset_name} and output dimensionality: {dimensionality} ==================")
            d = generate_standard_plot_and_stats_for_1d_data(metric_subset, show_plot=False)
            d["name"] = dimensionality
            dataset_stats.append(d)
        datasets_stats[dataset_name] = dataset_stats
    return datasets_stats


def get_time_statistics(datasets: Dict[str, pd.DataFrame], task_number: str, outputs_config: [str] = ['3', '5', '7']) -> dict:
    scaled_results = {}
    dependent_variable_name = "time_metric"
    datasets_names = list(datasets.keys())
    plt.figure()
    def _calculate_metric(row: pd.Series):
        if (row["btw_start_and_first_change"] == row["btw_first_and_last_slider_action"]) and (
                row["btw_start_and_first_change"] != 0):
            return row["btw_start_and_first_change"]
        elif (row["btw_start_and_first_change"] == row["btw_first_and_last_slider_action"]) and (
                row["btw_start_and_first_change"] == 0):
            return row["time_delta_seconds"]
        else:
            return row["btw_start_and_first_change"] + row["btw_first_and_last_slider_action"]
    for dataset_name, dataset_df in datasets.items():
        scaled_dfs = []
        for column, output_config in enumerate(outputs_config):
            temp_df =  dataset_df[dataset_df["number_of_outputs"] == output_config]
            temp_df[dependent_variable_name] = temp_df.apply(lambda row: _calculate_metric(row), axis=1)
            temp_df[dependent_variable_name] = (temp_df[dependent_variable_name] - temp_df[dependent_variable_name].min()) / (temp_df[dependent_variable_name].max() - temp_df[dependent_variable_name].min())
            scaled_dfs.extend(temp_df[dependent_variable_name].to_list())
        scaled_results[dataset_name] = scaled_dfs
    plt.boxplot(
        [
        scaled_results[datasets_names[i]] for i in range(len(datasets_names))
        ],
        showfliers=True, showmeans=True
    )
    plt.title(f"task {task_number}: normalized time till completion [s] for different datasets")
    plt.xticks([t + 1 for t in range(len(datasets_names))], datasets_names)
    plt.savefig(f"{current_dir}/task_visu_{task_number}_time_boxplot.png")
    # plt.show()
    res = check_norm_and_apply_stats_test(scaled_results)
    return res


def check_norm_and_apply_stats_test(dict_of_groups: {}, name: str = "standard") -> {}:
    dfs = list(dict_of_groups.values())
    payload = {}
    normality = True
    for d in dfs:
        if not is_data_normal(d):
            normality = False
    payload["are_populations_normal"] = normality
    if normality:
        are_means_different = is_the_mean_different(dfs)
        res = tukey_hsd(*dfs)
        payload["tukey_hsd"] = res
    else:
        are_means_different = is_the_mean_different(dfs, method="kruskal")
        res = posthoc_dunn(dfs, p_adjust='holm')
        payload["dunn"] = res
    payload["are_means_different"] = are_means_different
    return {name: payload}


def stats_avg_accuracy(datasets: pd.DataFrame, dependent_variable_name: str, task_name: str, outputs_config: [str] = ['3', '5', '7']) -> dict:
    scaled_results = {}
    datasets_names = list(datasets.keys())
    plt.figure()
    for dataset_name, dataset_df in datasets.items():
        conv_1_4_task_number = int(str(task_name)[-1])
        scaled_dfs = []
        for column, output_config in enumerate(outputs_config):
            dep_var_vals = dataset_df[dataset_df["number_of_outputs"] == output_config][dependent_variable_name].to_list()
            wfg_name = dataset_name.split("_")[0]
            starting_point = get_normalized_starting_point(wfg_name, int(output_config), conv_1_4_task_number)
            best_point = get_best_solution(wfg_name, int(output_config), conv_1_4_task_number)
            scaled_res_dataset = rescale_results(best=best_point, starting=starting_point, value=dep_var_vals,
                                        task_number=conv_1_4_task_number)
            scaled_dfs.extend(scaled_res_dataset)
        scaled_results[dataset_name] = scaled_dfs
    plt.boxplot(
        [
            scaled_results[datasets_names[i]] for i in range(len(datasets_names))
        ],
        showfliers=True, showmeans=True
    )
    plt.title(f"task {task_name}: normalized costs for different datasets")
    plt.xticks([t + 1 for t in range(len(datasets_names))], datasets_names)
    # plt.show()
    plt.savefig(f"{current_dir}/task_visu_{task_name}_avg_boxplot.png")
    res = check_norm_and_apply_stats_test(scaled_results)
    return res


def create_task_p_values_table(task_results:dict):
    """
    Create a table with the p-values for each task.
    :param task_results: dict with the results of each task
    :return: pd.DataFrame with the p-values for each task
    """
    accuracy_results = task_results["normalized_costs_task1"]
    task_p_values = {}
    for task, stats in accuracy_results.items():
        if stats["are_populations_normal"]:
            p_value = stats["tukey_hsd"].pvalue[0,1]
        else:
            p_value = stats["dunn"].iloc[0, 1]
        task_p_values[task] = p_value
    time_results = task_results["time_stats_task1"]
    time_task_p_values = {}
    for task, stats in time_results.items():
        if stats["are_populations_normal"]:
            p_value = stats["tukey_hsd"].pvalue[0,1]
        else:
            p_value = stats["dunn"].iloc[0, 1]
        time_task_p_values[task] = p_value
    return task_p_values


def main():
    df = read_data()
    eq_df, cvd_df, t1_df_merged, t2_df_merged, t3_df_merged, t4_df_merged, end_q_df = data_set_creator(df)
    common_users = get_user_intersection(t1_df_merged, t2_df_merged, t3_df_merged, t4_df_merged)
    t1_common, t2_common, t3_common, t4_common = common_members_dfs(t1_df_merged, t2_df_merged, t3_df_merged,
                                                                    t4_df_merged, common_users)
    dependent_variables_names_per_task = {
        1: "history_lowest_elec_cost",
        2: "history_lowest_overall_cost",
        3: "history_lowest_overall_cost",
        4: "history_highest_overall_cost"
    }
    stats = {}
    for task_number, task in enumerate([t1_common, t2_common, t3_common, t4_common]):
        task_stats = task_plotter(task, task_number=task_number+1, dependent_variable_name=dependent_variables_names_per_task[task_number+1])
        stats[f"task_{task_number+1}"] = task_stats
        # create_task_p_values_table(task_stats)
    print("Analysis complete.")
    # create_task_p_values_table(task4_stats)
    # plt.show()


if __name__ == '__main__':
    main()
