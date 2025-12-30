import pathlib
from helpers.study_helpers.general_study_helpers import *
from helpers.common_helpers import Tasks, DependentVariables

current_dir = pathlib.Path(__file__).parent.absolute()

DATASETS = ["WFG2", "WFG4", "WFG5"]
OUPUTS_CONFIG = ['3', '5', '7']

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
    _print_latex_table_row(payload, "normalized_costs", task)
    _print_latex_table_row(payload, "time_stats", task)


def _print_latex_table_row(statistics_of_interest: dict, task_dependent_variable_name: str, task: str) -> None:
    stats = statistics_of_interest[task_dependent_variable_name]
    configs = list(stats.keys())
    cells = []
    key = ("line_charts", "colorfields")
    for dataset in DATASETS:
        for output in OUPUTS_CONFIG:
            config_suffix = f"{dataset}_{output}"
            matched_config = [c for c in configs if config_suffix in c][0]
            print(f"Matched config: {matched_config}")
            metric = stats[matched_config]['standard']['confidence_intervals_viz'][key]['metric']
            print(f"Metric name: {stats[matched_config]['standard']['confidence_intervals_viz'][key]['metric_name']}")
            ci = stats[matched_config]['standard']['confidence_intervals_viz'][key]['ci']
            cell = f"{metric:.2f}{ci}"
            cells.append(cell)
            i = configs.index(matched_config)
            configs.pop(i)
        dataset_config =  [c for c in configs if dataset in c][0]
        print(f"Dataset config: {dataset_config}")
        metric = stats[dataset_config]['standard']['confidence_intervals_viz'][key]['metric']
        print(f"Metric name: {stats[dataset_config]['standard']['confidence_intervals_viz'][key]['metric_name']}")
        ci = stats[dataset_config]['standard']['confidence_intervals_viz'][key]['ci']
        cell = f"{metric:.2f}{ci}"
        cells.append(cell)
    for output in OUPUTS_CONFIG:
        config_suffix = f"_{output}"
        matched_config = [c for c in configs if config_suffix in c][0]
        print(f"Matched config: {matched_config}")
        metric = stats[matched_config]['standard']['confidence_intervals_viz'][key]['metric']
        print(f"Metric name: {stats[matched_config]['standard']['confidence_intervals_viz'][key]['metric_name']}")
        ci = stats[matched_config]['standard']['confidence_intervals_viz'][key]['ci']
        cell = f"{metric:.2f}{ci}"
        cells.append(cell)
        i = configs.index(matched_config)
        configs.pop(i)
    task_metric = stats[f"{task}"]['standard']["confidence_intervals_viz"][key]['metric']
    print(f"Task Metric name: {stats[f'{task}']['standard']['confidence_intervals_viz'][key]['metric_name']}")
    task_ci = stats[f"{task}"]['standard']["confidence_intervals_viz"][key]['ci']
    task_cell = f"{task_metric:.2f}{task_ci}"
    cells.append(task_cell)
    print(f"""Confidence intervals for {task}""")
    table_row = "&"
    for i, cell in enumerate(cells):
        table_row += f"{cell} "
        if (i+1) % 4 == 0:
            table_row += " \\\\ & \n"
        else:
            table_row += "& "
    table_row = table_row.replace(")", "]").replace("(", "[")
    print(f"LaTeX table row for task {task} and dependent variable {task_dependent_variable_name}: {table_row} \\\\")


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
        print_effect_sizes_with_ci_as_table(stats, task_number)
        print(f"Completed analysis for task {i+1}")
        results[f"task_{i+1}"] = stats


if __name__ == '__main__':
    main()
