import pathlib
from helpers.study_helpers.general_study_helpers import *
from helpers.common_helpers import Tasks, DependentVariables

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


def task_analysis(df:pd.DataFrame, task_name:str, dependent_variable_name:str):
    line_chart_df = df[df["visualization"] == "line_chart"]
    heatmap_df = df[df["visualization"] == "heatmap"]
    get_1d_stats_on_experiment(line_chart_df, heatmap_df, dependent_variable_name, task_name)
    means_stats_payload = stats_means_accuracy(df, dependent_variable_name)
    time_stats_payload = stats_time(df)
    full_stats = {f"normalized_costs": means_stats_payload,
                  f"time_stats": time_stats_payload,
                  "task_number": task_name}
    return full_stats


def task_1_plotter(t0_df_merged: pd.DataFrame):
    line_chart_df = t0_df_merged[t0_df_merged["visualization"] == "line_chart"]
    heatmap_df = t0_df_merged[t0_df_merged["visualization"] == "heatmap"]
    dependent_variable_name = "history_lowest_elec_cost"
    task_number = [i for i in list(locals().keys()) if i.__contains__("_df_merged")][0].split("_df_merged")[0].split("t")[1]
    lc_stats, h_stats = get_1d_stats_on_experiment(line_chart_df, heatmap_df, dependent_variable_name, task_number)
    # these are all the statistical tests
    # the statistics for each of the 1d data are stored in lc_stats for line_charts and in h_stats for heatmaps
    means_stats_payload = stats_means_accuracy(t0_df_merged, dependent_variable_name)
    time_stats_payload = stats_time(t0_df_merged)
    full_stats = {"normalized_costs_task1": means_stats_payload,
                  "time_stats_task1": time_stats_payload}
    return full_stats


def task_2_plotter(t1_df_merged: pd.DataFrame):
    line_chart_df = t1_df_merged[t1_df_merged["visualization"] == "line_chart"]
    heatmap_df = t1_df_merged[t1_df_merged["visualization"] == "heatmap"]
    dependent_variable_name = "history_lowest_overall_cost"
    task_number = [i for i in list(locals().keys()) if i.__contains__("_df_merged")][0].split("_df_merged")[0].split("t")[1]
    lc_stats, h_stats = get_1d_stats_on_experiment(line_chart_df, heatmap_df, dependent_variable_name, task_number)
    means_stats_payload = stats_means_accuracy(t1_df_merged, dependent_variable_name)
    time_stats_payload = stats_time(t1_df_merged)
    full_stats = {"normalized_costs_task2": means_stats_payload,
                  "time_stats_task2": time_stats_payload}
    return full_stats


def task_3_plotter(t2_df_merged: pd.DataFrame):
    line_chart_df = t2_df_merged[t2_df_merged["visualization"] == "line_chart"]
    heatmap_df = t2_df_merged[t2_df_merged["visualization"] == "heatmap"]
    dependent_variable_name = "history_lowest_overall_cost"
    task_number = [i for i in list(locals().keys()) if i.__contains__("_df_merged")][0].split("_df_merged")[0].split("t")[1]
    lc_stats, h_stats = get_1d_stats_on_experiment(line_chart_df, heatmap_df, dependent_variable_name, task_number)
    means_stats_payload = stats_means_accuracy(t2_df_merged, dependent_variable_name)
    time_stats_payload = stats_time(t2_df_merged)
    full_stats = {"normalized_costs_task3": means_stats_payload,
                  "time_stats_task3": time_stats_payload}
    return full_stats


def task_4_plotter(t3_df_merged: pd.DataFrame):
    line_chart_df = t3_df_merged[t3_df_merged["visualization"] == "line_chart"]
    heatmap_df = t3_df_merged[t3_df_merged["visualization"] == "heatmap"]
    dependent_variable_name = "history_highest_overall_cost"
    task_number = [i for i in list(locals().keys()) if i.__contains__("_df_merged")][0].split("_df_merged")[0].split("t")[1]
    lc_stats, h_stats = get_1d_stats_on_experiment(line_chart_df, heatmap_df, dependent_variable_name, task_number)
    means_stats_payload = stats_means_accuracy(t3_df_merged,  dependent_variable_name)
    time_stats_payload = stats_time(t3_df_merged)
    full_stats = {"normalized_costs_task4": means_stats_payload,
                  "time_stats_task4": time_stats_payload}
    return full_stats


def create_task_p_values_table(task_results:dict):
    """
    Create a table with the p-values for each task.
    :param task_results: dict with the results of each task
    :return: pd.DataFrame with the p-values for each task
    """
    accuracy_results = task_results["normalized_costs_task1"]
    task_p_values = {}
    for task, stats in accuracy_results.items():
        if stats["standard"]["are_populations_normal"]:
            p_value = stats["standard"]["tukey_hsd"].pvalue[0,1]
        else:
            p_value = stats["standard"]["dunn"].iloc[0, 1]
        task_p_values[task] = p_value
    time_results = task_results["time_stats_task1"]
    time_task_p_values = {}
    for task, stats in time_results.items():
        if stats["standard"]["are_populations_normal"]:
            p_value = stats["standard"]["tukey_hsd"].pvalue[0,1]
        else:
            p_value = stats["standard"]["dunn"].iloc[0, 1]
        time_task_p_values[task] = p_value
    return task_p_values

def print_effect_sizes_with_ci_as_table(payload: dict, task: str) -> None:
    cost_task_dependent_variable_name = {
        task: f"normalized_costs_{task}",
    }
    _print_latex_table_row(payload, cost_task_dependent_variable_name, task)
    time_task_dependent_variable_name = {
        task: f"time_stats_{task}",
    }
    _print_latex_table_row(payload, time_task_dependent_variable_name, task)


def _print_latex_table_row(statistics_of_interest: dict, task_dependent_variable_name: str, task: str) -> None:
    stats = statistics_of_interest[task_dependent_variable_name[task]]
    cell11 = f"{stats['effect_size_lc']['metric']:.2f}"
    cell12 = f"{stats['effect_size_h']['metric']:.2f}"
    ci_lc_keys = list(stats['confidence_intervals_lc'].keys())
    ci_h_keys = list(stats['confidence_intervals_h'].keys())
    cell13 = f"{stats['confidence_intervals_lc'][ci_lc_keys[0]]['metric']:.2f}{stats['confidence_intervals_lc'][ci_lc_keys[0]]['ci']}"
    cell14 = f"{stats['confidence_intervals_lc'][ci_lc_keys[1]]['metric']:.2f}{stats['confidence_intervals_lc'][ci_lc_keys[1]]['ci']}"
    cell15 = f"{stats['confidence_intervals_lc'][ci_lc_keys[2]]['metric']:.2f}{stats['confidence_intervals_lc'][ci_lc_keys[2]]['ci']}"
    cell16 = f"{stats['confidence_intervals_h'][ci_h_keys[0]]['metric']:.2f}{stats['confidence_intervals_h'][ci_h_keys[0]]['ci']}"
    cell17 = f"{stats['confidence_intervals_h'][ci_h_keys[1]]['metric']:.2f}{stats['confidence_intervals_h'][ci_h_keys[1]]['ci']}"
    cell18 = f"{stats['confidence_intervals_h'][ci_h_keys[2]]['metric']:.2f}{stats['confidence_intervals_h'][ci_h_keys[2]]['ci']}"
    cell19 = f"{stats['confidence_intervals_h'][ci_h_keys[3]]['metric']:.2f}{stats['confidence_intervals_h'][ci_h_keys[3]]['ci']}"
    cell110 = f"{stats['confidence_intervals_h'][ci_h_keys[4]]['metric']:.2f}{stats['confidence_intervals_h'][ci_h_keys[4]]['ci']}"
    cell111 = f"{stats['confidence_intervals_h'][ci_h_keys[5]]['metric']:.2f}{stats['confidence_intervals_h'][ci_h_keys[5]]['ci']}"
    cell112 = f"{stats['confidence_intervals_h'][ci_h_keys[6]]['metric']:.2f}{stats['confidence_intervals_h'][ci_h_keys[6]]['ci']}"
    cell113 = f"{stats['confidence_intervals_h'][ci_h_keys[7]]['metric']:.2f}{stats['confidence_intervals_h'][ci_h_keys[7]]['ci']}"
    cell114 = f"{stats['confidence_intervals_h'][ci_h_keys[8]]['metric']:.2f}{stats['confidence_intervals_h'][ci_h_keys[8]]['ci']}"
    cell115 = f"{stats['confidence_intervals_h'][ci_h_keys[9]]['metric']:.2f}{stats['confidence_intervals_h'][ci_h_keys[9]]['ci']}"
    cell116 = f"{stats['confidence_intervals_h'][ci_h_keys[10]]['metric']:.2f}{stats['confidence_intervals_h'][ci_h_keys[10]]['ci']}"
    cell117 = f"{stats['confidence_intervals_h'][ci_h_keys[11]]['metric']:.2f}{stats['confidence_intervals_h'][ci_h_keys[11]]['ci']}"
    cell118 = f"{stats['confidence_intervals_h'][ci_h_keys[12]]['metric']:.2f}{stats['confidence_intervals_h'][ci_h_keys[12]]['ci']}"
    cell119 = f"{stats['confidence_intervals_h'][ci_h_keys[13]]['metric']:.2f}{stats['confidence_intervals_h'][ci_h_keys[13]]['ci']}"
    cell120 = f"{stats['confidence_intervals_h'][ci_h_keys[14]]['metric']:.2f}{stats['confidence_intervals_h'][ci_h_keys[14]]['ci']}"
    cells = [
        cell11, " & ", cell12, " & ", cell13, " & ", cell14,
        " & ", cell15, " & ", cell16, " & ", cell17, " & ", cell18 ,
        " & ", cell19, " & ", cell110, " & ", cell111, " & ", cell112 ,
        " & ", cell113, " & ", cell114, " & ", cell115, " & ", cell116 ,
        " & ", cell117, " & ", cell118, " & ", cell119, " & ", cell120
    ]
    print(f"""Effect sizes and confidence intervals for {task}""")
    table_row = str.join("", cells)
    table_row = table_row.replace(")", "]").replace("(", "[")
    print(f"LaTeX table row for task {task}: & {table_row} \\\\")



def main():
    df = read_data()
    eq_df, cvd_df, t1_df_merged, t2_df_merged, t3_df_merged, t4_df_merged, end_q_df = data_set_creator(df)
    common_users = get_user_intersection(t1_df_merged, t2_df_merged, t3_df_merged, t4_df_merged)
    t1_common, t2_common, t3_common, t4_common = common_members_dfs(t1_df_merged, t2_df_merged, t3_df_merged,
                                                                    t4_df_merged, common_users)
    datasets = [t1_common, t2_common, t3_common, t4_common]
    dependent_variable_names = [
        DependentVariables.HISTORY_LOWEST_ELEC_COST.value,
        DependentVariables.HISTORY_LOWEST_OVERALL_COST.value,
        DependentVariables.HISTORY_LOWEST_OVERALL_COST.value,
        DependentVariables.HISTORY_HIGHEST_OVERALL_COST.value,
    ]
    tasks = [t.value for t in Tasks]
    results = {}
    for i, task_number in enumerate(tasks):
        print(f"Starting analysis for task {i+1}")
        stats = task_analysis(datasets[i], task_number, dependent_variable_name=dependent_variable_names[i])
        # stats = plotters[i](datasets[i])
        # print_effect_sizes_with_ci_as_table(stats, task_number)
        print(f"Completed analysis for task {i+1}")
        results[f"task_{i+1}"] = stats


if __name__ == '__main__':
    main()
