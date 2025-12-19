import pandas as pd

from helpers.height_pilot_helpers import *


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


def task_1_plotter(t1_df_merged: pd.DataFrame) -> dict:
    """
    Plots the data for task 1 and return the statistics for each height and visualization for both costs and time
    :param t1_df_merged: data for task 1
    :return: statistics as a dictionary
    """
    dependent_variable_name = "history_lowest_elec_cost"
    # t1_df_merged = normalize_data(t1_df_merged, dependent_variable_name, "task1")
    t1_df_merged.dropna(subset=["history_lowest_elec_cost"], inplace=True)
    line_chart_df = t1_df_merged[t1_df_merged["visualization"] == "line_chart"]
    colorfield_df = t1_df_merged[t1_df_merged["visualization"] == "heatmap"]
    # generates some 1D descriptive statistics for each height group
    task_number = [i for i in list(locals().keys()) if i.__contains__("_df_merged")][0].split("_df_merged")[0].split("t")[1]
    lc_stats, h_stats = get_1D_statistics_for_each_height(colorfield_df, line_chart_df, dependent_variable_name, int(task_number))
    # calculates the statistics for each height and visualization as in the dissertation and potentially future papers
    means_stats_payload = stats_means_accuracy_height(t1_df_merged, dependent_variable_name)
    stats_1d = {"line_chart": lc_stats, "colorfields": h_stats}
    means_stats_payload["stats_1d"] = stats_1d
    viz_stats_payload = stats_means_accuracy_viz(t1_df_merged, dependent_variable_name)
    time_height_stats_payload = stats_means_time_height(t1_df_merged)
    viz_time_stats_payload = stats_means_time_viz(t1_df_merged)
    boxplots(colorfield_df, line_chart_df, dependent_variable_name, t1_df_merged)
    full_stats = {"cost_means_stats": {
        "elec_cost_by_height_stats": means_stats_payload,
        "elec_costs_by_viz_lc_or_cf": viz_stats_payload,
    },
        "time_means_stats": {
            "time_to_execute_by_height": time_height_stats_payload,
            "time_by_viz_lc_or_h": viz_time_stats_payload,
        }}
    return full_stats


def task_2_plotter(t2_df_merged: pd.DataFrame) -> dict:
    """
    Plots the data for task 2 and return the statistics for each height and visualization for both costs and time
    :param t2_df_merged: data for task 2
    :return: statistics as a dictionary
    """
    dependent_variable_name = "history_lowest_overall_cost"
    # t2_df_merged = normalize_data(t2_df_merged, dependent_variable_name, "task2")
    line_chart_df = t2_df_merged[t2_df_merged["visualization"] == "line_chart"]
    colorfield_df = t2_df_merged[t2_df_merged["visualization"] == "heatmap"]
    lc_stats = []
    # generates some 1D descriptive statistics for each height group
    task_number = [i for i in list(locals().keys()) if i.__contains__("_df_merged")][0].split("_df_merged")[0].split("t")[1]
    lc_stats, h_stats = get_1D_statistics_for_each_height(colorfield_df, line_chart_df, dependent_variable_name, int(task_number))
    # calculates the statistics for each height and visualization as in the dissertation and potentially future papers
    means_stats_payload = stats_means_accuracy_height(t2_df_merged, dependent_variable_name)
    stats_1d = {"line_chart": lc_stats, "heatmap": h_stats}
    means_stats_payload["stats_1d"] = stats_1d
    viz_stats_payload = stats_means_accuracy_viz(t2_df_merged, dependent_variable_name)
    time_height_stats_payload = stats_means_time_height(t2_df_merged)
    viz_time_stats_payload = stats_means_time_viz(t2_df_merged)
    boxplots(colorfield_df, line_chart_df, dependent_variable_name, t2_df_merged)
    full_stats = {"cost_means_stats": {
        "total_costs_by_height_stats": means_stats_payload,
        "total_costs_by_viz_lc_or_hm": viz_stats_payload,
    },
        "time_means_stats": {
            "time_to_execute_by_height": time_height_stats_payload,
            "time_by_viz_lc_or_h": viz_time_stats_payload,
        }}
    return full_stats


def task_3_plotter(t3_df_merged: pd.DataFrame) -> dict:
    """
    Plots the data for task 3 and return the statistics for each height and visualization for both costs and time
    :param t3_df_merged: data for task 3
    :return: statistics as a dictionary
    """
    dependent_variable_name = "history_lowest_overall_cost"
    # t3_df_merged = normalize_data(t3_df_merged, dependent_variable_name, "task3")
    line_chart_df = t3_df_merged[t3_df_merged["visualization"] == "line_chart"]
    colorfield_df = t3_df_merged[t3_df_merged["visualization"] == "heatmap"]
    lc_stats = []
    # generates some 1D descriptive statistics for each height group
    task_number = [i for i in list(locals().keys()) if i.__contains__("_df_merged")][0].split("_df_merged")[0].split("t")[1]
    lc_stats, h_stats = get_1D_statistics_for_each_height(colorfield_df, line_chart_df, dependent_variable_name, int(task_number))
    # calculates the statistics for each height and visualization as in the dissertation and potentially future papers
    means_stats_payload = stats_means_accuracy_height(t3_df_merged, "history_lowest_overall_cost")
    stats_1d = {"line_chart": lc_stats, "heatmap": h_stats}
    means_stats_payload["stats_1d"] = stats_1d
    viz_stats_payload = stats_means_accuracy_viz(t3_df_merged, "history_lowest_overall_cost")
    time_height_stats_payload = stats_means_time_height(t3_df_merged)
    viz_time_stats_payload = stats_means_time_viz(t3_df_merged)
    boxplots(colorfield_df, line_chart_df, dependent_variable_name, t3_df_merged)
    full_stats = {"cost_means_stats": {
        "total_costs_by_height_stats": means_stats_payload,
        "total_costs_by_viz_lc_or_hm": viz_stats_payload,
    },
        "time_means_stats": {
            "time_to_execute_by_height": time_height_stats_payload,
            "time_by_viz_lc_or_h": viz_time_stats_payload,
        }}
    return full_stats


def task_4_plotter(t4_df_merged: pd.DataFrame) -> dict:
    dependent_variable_name = "history_highest_overall_cost"
    # t4_df_merged = normalize_data(t4_df_merged, dependent_variable_name, "task4")
    line_chart_df = t4_df_merged[t4_df_merged["visualization"] == "line_chart"]
    colorfield_df = t4_df_merged[t4_df_merged["visualization"] == "heatmap"]
    lc_stats = []
    # generates some 1D descriptive statistics for each height group
    task_number = [i for i in list(locals().keys()) if i.__contains__("_df_merged")][0].split("_df_merged")[0].split("t")[1]
    lc_stats, h_stats = get_1D_statistics_for_each_height(colorfield_df, line_chart_df, dependent_variable_name, int(task_number))
    # calculates the statistics for each height and visualization as in the dissertation and potentially future papers
    means_accuracy_height_stats_payload = stats_means_accuracy_height(t4_df_merged, "history_highest_overall_cost")
    stats_1d = {"line_chart": lc_stats, "heatmap": h_stats}
    means_accuracy_height_stats_payload["stats_1d"] = stats_1d
    means_accuracy_viz_stats_payload = stats_means_accuracy_viz(t4_df_merged, "history_highest_overall_cost")
    time_height_stats_payload = stats_means_time_height(t4_df_merged)
    viz_time_stats_payload = stats_means_time_viz(t4_df_merged)
    boxplots(colorfield_df, line_chart_df, dependent_variable_name, t4_df_merged)
    full_stats = {"cost_means_stats": {
        "total_costs_by_height_stats": means_accuracy_height_stats_payload,
        "total_costs_by_viz_lc_or_hm": means_accuracy_viz_stats_payload,
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
    cost_task_dependent_variable_name = {
        "task1": "elec_cost_by_height_stats",
        "task2": "total_costs_by_height_stats",
        "task3": "total_costs_by_height_stats",
        "task4": "total_costs_by_height_stats",
    }
    cost_data = payload["cost_means_stats"]
    _print_latex_table_row(cost_data, cost_task_dependent_variable_name, task, "cost")
    time_data = payload["time_means_stats"]
    time_task_dependent_variable_name = {
        "task1": "time_to_execute_by_height",
        "task2": "time_to_execute_by_height",
        "task3": "time_to_execute_by_height",
        "task4": "time_to_execute_by_height",
    }
    _print_latex_table_row(time_data, time_task_dependent_variable_name, task, "time")


def _print_latex_table_row(statistics_of_interest: dict, task_dependent_variable_name: str, task: str, metric_name: str) -> None:
    stats = statistics_of_interest[task_dependent_variable_name[task]]
    cell11 = f"{stats['effect_size_lc']['metric']:.2f}"
    cell12 = f"{stats['effect_size_h']['metric']:.2f}"
    ci_keys = list(stats['confidence_intervals_lc'].keys())
    cell13 = f"{stats['confidence_intervals_lc'][ci_keys[0]]['metric']:.2f}{stats['confidence_intervals_lc'][ci_keys[0]]['ci']}"
    cell14 = f"{stats['confidence_intervals_lc'][ci_keys[1]]['metric']:.2f}{stats['confidence_intervals_lc'][ci_keys[1]]['ci']}"
    cell15 = f"{stats['confidence_intervals_lc'][ci_keys[2]]['metric']:.2f}{stats['confidence_intervals_lc'][ci_keys[2]]['ci']}"
    cell16 = f"{stats['confidence_intervals_h'][ci_keys[0]]['metric']:.2f}{stats['confidence_intervals_h'][ci_keys[0]]['ci']}"
    cell17 = f"{stats['confidence_intervals_h'][ci_keys[1]]['metric']:.2f}{stats['confidence_intervals_h'][ci_keys[1]]['ci']}"
    cell18 = f"{stats['confidence_intervals_h'][ci_keys[2]]['metric']:.2f}{stats['confidence_intervals_h'][ci_keys[2]]['ci']}"
    cells = [cell11, " & ", cell12, " & ", cell13, " & ", cell14, " & ", cell15, " & ", cell16, " & ", cell17, " & ",
             cell18]
    print(f"""Effect sizes and confidence intervals for {task}""")
    table_row = str.join("", cells)
    table_row = table_row.replace(")", "]").replace("(", "[")
    print(f"LaTeX table row for task {task} and metric name {metric_name}: & {table_row} \\\\")

def main():
    data = read_inputs()
    tasks = ["task1", "task2", "task3", "task4"]
    plotters = [task_1_plotter, task_2_plotter, task_3_plotter,task_4_plotter]
    results = {}
    for index, task_number in enumerate(tasks):
        concat_df = concater(data[task_number]["gy"], data[task_number]["list"])
        plotter = plotters[index]
        result = plotter(concat_df)
        print_effect_sizes_with_ci_as_table(result, task_number)
        results[task_number] = result
        print(f"""results for {task_number} are: 
        {result}""")
    print("All tasks processed.")


if __name__ == '__main__':
    main()

