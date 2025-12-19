import pandas as pd
from helpers.height_pilot_helpers import *
from helpers.common_helpers import DependentVariablePerTask, Tasks


def read_inputs() -> dict:
    """
    reads all the data from csv files for this pilot. Data was cleaned for users which failed the attention or comprehension checks
    :return: dictionary with the participants' answers by organization
    """
    base_path = "data/height_pilot_data/"
    tasks = ["task1", "task2", "task3", "task4"]
    data = {}
    for index, task in enumerate(tasks):
        gy_part = pd.read_csv(base_path + "gy_t" + str(index) + "_cleaned.csv")
        list_part = pd.read_csv(base_path + "list_t" + str(index) + "_cleaned.csv")
        data[task] = {"gy": gy_part, "list": list_part}
    return data


# obtained by breakpointing on the ./helpers/general_study_helpers.py (initiated by study_analyzer.py) on function stats_means_accuracy
DATA_EXTREMA = {
"WFG2": {
    "task1": {
        "maximum": 0.833,
        "minimum": 0.074
    },
    "task2": {
        "maximum": 2.042,
        "minimum": 1.22
    },
    "task3": {
        "maximum": 2.042,
        "minimum": 0.266
    },
    "task4": {
        "minimum": 0.28,
        "maximum": 1.025
    }
}}

def task_ploter(
        t_data: pd.DataFrame,
        dependent_variable_name: DependentVariablePerTask,
        task_name: Tasks
) -> dict:
    """
    Plots the data for a given task and return the statistics for each height and visualization for both costs and time
    :param t_data:  data for the given task
    :param dependent_variable_name:
    :param task_number:
    :return:
    """
    # t1_df_merged = normalize_data(t1_df_merged, dependent_variable_name, "task1")
    task_number = int(task_name[-1])
    t_data.dropna(subset=[dependent_variable_name], inplace=True)
    line_chart_df = t_data[t_data["visualization"] == "line_chart"]
    colorfield_df = t_data[t_data["visualization"] == "heatmap"]
    # generates some 1D descriptive statistics for each height group
    lc_stats, h_stats = get_1D_statistics_for_each_height(colorfield_df, line_chart_df, dependent_variable_name,
                                                          int(task_number))
    # calculates the statistics for each height and visualization as in the dissertation and potentially future papers
    means_stats_payload = stats_means_accuracy_height(t_data, dependent_variable_name, HEIGHTS)
    stats_1d = {"line_chart": lc_stats, "colorfields": h_stats}
    means_stats_payload["stats_1d"] = stats_1d
    viz_stats_payload = stats_means_accuracy_viz(t_data, dependent_variable_name)
    time_height_stats_payload = stats_means_time_height(t_data, heights=HEIGHTS)
    viz_time_stats_payload = stats_means_time_viz(t_data)
    boxplots(colorfield_df, line_chart_df, dependent_variable_name, t_data)
    full_stats = {"cost_means_stats": {
        "cost_by_height": means_stats_payload,
        "cost_by_viz": viz_stats_payload,
    },
        "time_means_stats": {
            "time_to_execute_by_height": time_height_stats_payload,
            "time_by_viz_lc_or_h": viz_time_stats_payload,
        }}
    return full_stats

def normalize_data(df:pd.DataFrame, column_name:str, task_number:str) -> pd.DataFrame:
    maximum =  DATA_EXTREMA["WFG2"][task_number]["maximum"]
    minimum = DATA_EXTREMA["WFG2"][task_number]["minimum"]
    df[column_name] = (df[column_name] - minimum) / (maximum - minimum)
    # authorizing 5% of error on each side due to potential floating point errors
    if df[column_name].max() > 1.05 or df[column_name].min() < -0.05:
        raise ValueError("Normalization failed, data out of bounds after normalization")
    def _apply_normalization_bounds(value:float) -> float:
        if value > 1.0:
            print("value greater than 1.0 detected, setting to 1.0")
            return 1.0
        elif value < 0.0:
            print("value less than 0.0 detected, setting to 0.0")
            return 0.0
        else:
            return value
    df[column_name] = df[column_name].apply(lambda x: _apply_normalization_bounds(x))
    return df


def print_effect_sizes_with_ci_as_table(payload: dict, task: str) -> None:
    cost_data = payload["cost_means_stats"]
    _print_latex_table_row(cost_data, "cost_by_height", task, "cost", factor_levels=["lc", "h"])
    # _print_latex_table_row(cost_data, "cost_by_height", task, "cost", factor_levels=["lc", "h"])
    time_data = payload["time_means_stats"]
    _print_latex_table_row(time_data, "time_to_execute_by_height", task, "time", factor_levels=["lc", "h"])
    # _print_latex_table_row(time_data, "time_to_execute_by_height", task, "time", factor_levels=["lc", "h"])


def _print_latex_table_row(
        statistics_of_interest: dict,
        stat_name: str,
        task: str,
        metric_name: str,
        factor_levels: list[str]
) -> None:
    stats = statistics_of_interest[stat_name]
    cells = []
    for factor_level in factor_levels:
        cells.append(f"{stats[f'effect_size_{factor_level}']['metric']:.2f}")
    for i, factor_level in enumerate(factor_levels):
        ci_keys = list(stats[f'confidence_intervals_{factor_level}'].keys())
        for j, key in enumerate(ci_keys):
            cells.append(f"{stats[f'confidence_intervals_{factor_level}'][ci_keys[j]]['metric']:.2f}{stats[f'confidence_intervals_{factor_level}'][ci_keys[j]]['ci']}")
    print(f"""Effect sizes and confidence intervals for {task}""")
    table_row = str.join(" & ", cells)
    table_row = table_row.replace(")", "]").replace("(", "[")
    print(f"LaTeX table row for task {task} and metric name {metric_name}: & {table_row} \\\\")

def main():
    data = read_inputs()
    tasks = [task.value for task in Tasks]
    results = {}
    for index, task_name in enumerate(tasks):
        concat_df = concater(data[task_name]["gy"], data[task_name]["list"])
        result = task_ploter(concat_df, DependentVariablePerTask[task_name].value, task_name)
        print_effect_sizes_with_ci_as_table(result, task_name)
        results[task_name] = result
        print(f"""results for {task_name} are: 
        {result}""")
    print("All tasks processed.")


if __name__ == '__main__':
    main()

