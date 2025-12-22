from helpers.colormap_pilot_helpers import *
from helpers.common_helpers import *

def read_data():
    df = pd.read_csv("data/colormap_pilot_data/colormapPilot_all.csv", delimiter="|")
    df = clean_raw_df(df)
    df.columns = ["userID", "stepName", "stepValue", "studyID", "sessionID", "timeBackend"]
    df = remove_invalid_submissions(df)
    return df


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

def task_plotter(t_data:pd.DataFrame, dependent_variable_name:DependentVariablePerTask, task_name:Tasks) -> dict:
    line_chart_df = t_data[t_data["visualization"] == "line_chart"]
    heatmap_df = t_data[t_data["visualization"] == "heatmap"]
    task_number = int(task_name[-1])
    lc_stats, h_stats = get_1d_stats_on_colormaps(line_chart_df, heatmap_df, dependent_variable_name, task_number)
    # the statistics for each of the 1d data are stored in lc_stats for line_charts and in h_stats for heatmaps
    means_stats_payload = stats_means_accuracy_colormaps(t_data, dependent_variable_name)
    stats_1d = {"line_chart": lc_stats, "heatmap": h_stats}
    means_stats_payload["stats_1d"] = stats_1d
    viz_stats_payload = stats_means_accuracy_viz(t_data, dependent_variable_name)
    time_colormaps_stats_payload = stats_means_time_colormap(t_data)
    time_viz_stats_payload = stats_means_time_viz(t_data)
    means_quali_seq_stats_payload = stats_means_sequential_qualitative_accuracy(t_data, dependent_variable_name)
    time_quali_seq_stats_payload = stats_means_sequential_qualitative_time(t_data, dependent_variable_name)
    # here statistical results stop
    line_chart_df.boxplot(column=dependent_variable_name, by=["colormap"], showmeans=True)
    plt.title("line_chart")
    plt.ylabel(f"{dependent_variable_name} [-]")
    heatmap_df["colormap"] = pd.Categorical(heatmap_df["colormap"], categories=["i_want_hue", "mokole", "set1", "oranges", "purple2", "yellow_red"])
    heatmap_df.sort_values("colormap", inplace=True)
    heatmap_df.boxplot(column=f"{dependent_variable_name}", by=["colormap"])
    plt.ylabel(f"{dependent_variable_name} [-]")
    plt.title("colorfield")
    plt.show()
    t_data.boxplot(column=f"{dependent_variable_name}", by=["visualization"])
    plt.ylabel(f"{dependent_variable_name} [-]")
    plt.title("line_charts vs colorfields")
    plt.show()
    print(f"===============DONE in for task 1==================")
    full_stats = {"cost_means_stats": {
        "costs_by_colormap": means_stats_payload,
        "costs_by_viz": viz_stats_payload,
        "costs_by_seq_or_quali": means_quali_seq_stats_payload
    },
        "time_means_stats": {
            "time_by_colormap": time_colormaps_stats_payload,
            "time_by_viz": time_viz_stats_payload,
            "time_by_seq_or_quali": time_quali_seq_stats_payload
        }}
    return full_stats

def print_effect_sizes_with_ci_as_table(payload: dict, task: str) -> None:
    cost_data = payload["cost_means_stats"]
    _print_detailed_latex_table_row(cost_data, "costs_by_colormap", task, "cost_metric")
    _print_summary_latex_table_row(cost_data, task, "cost", factor_levels=["lc", "h"],
                                   global_stats_name="costs_by_colormap", viz_stats_name="costs_by_viz", seq_stats_name="costs_by_seq_or_quali")
    time_data = payload["time_means_stats"]
    _print_detailed_latex_table_row(time_data, "time_by_colormap", task, "time_metric")
    _print_summary_latex_table_row(time_data, task, "time", factor_levels=["lc", "h"],
                                   global_stats_name="time_by_colormap", viz_stats_name="time_by_viz", seq_stats_name="time_by_seq_or_quali")


def _print_detailed_latex_table_row(statistics_of_interest: dict, stat_name: str, task: str, metric_name: str, ) -> None:
    stats = statistics_of_interest[stat_name]
    cells = []
    for viz in VisualizationTypes:
        cells.append( f"{stats['effect_size_'+viz.value]['metric']:.2f}")
    ci_lc_keys = list(stats['confidence_intervals_lc'].keys())
    ci_h_keys = list(stats['confidence_intervals_h'].keys())
    for index, key in enumerate(ci_lc_keys):
        cells.append(f"{stats['confidence_intervals_lc'][ci_lc_keys[index]]['metric']:.2f}{stats['confidence_intervals_lc'][ci_lc_keys[index]]['ci']}")
    for index, key in enumerate(ci_h_keys):
        cells.append(f"{stats['confidence_intervals_h'][ci_h_keys[index]]['metric']:.2f}{stats['confidence_intervals_h'][ci_h_keys[index]]['ci']}")
    print(f"""Effect sizes and confidence intervals for {task}""")
    table_row = str.join(" & ", cells)
    table_row = table_row.replace(")", "]").replace("(", "[")
    print(f"LaTeX table row for task {task} and metric name {metric_name}: & {table_row} \\\\")


def _print_summary_latex_table_row(
        statistics_of_interest: dict,
        task: str,
        metric_name: str,
        factor_levels: list[str],
        global_stats_name: str = "costs_by_colormap",
        viz_stats_name: str = "costs_by_viz",
        seq_stats_name: str = "cost_by_seq_or_quali"

) -> None:
    global_stats = statistics_of_interest[global_stats_name]
    cells = []
    for factor_level in factor_levels:
        print(f"cell ES_{factor_level}: {global_stats[f'effect_size_{factor_level}']}")
        cells.append(f"{global_stats[f'effect_size_{factor_level}']['metric']:.2f}")
    cost_by_seq = statistics_of_interest[seq_stats_name]
    print(f"cell ES_seq_or_quali: {cost_by_seq['confidence_intervals_colormap_type']}")
    pairwise_seq_metric = cost_by_seq["confidence_intervals_colormap_type"][("sequential", "qualitative")]["metric"]
    pairwise_seq_ci = cost_by_seq["confidence_intervals_colormap_type"][("sequential", "qualitative")]["ci"]
    cells.append(f"{pairwise_seq_metric:.2f}{pairwise_seq_ci}")
    cost_by_viz = statistics_of_interest[viz_stats_name]
    # print(f"cell ES_viz: {cost_by_viz['effect_size_viz']}")
    # global_viz = cost_by_viz["effect_size_viz"]["metric"]
    # cells.append(f"{global_viz:.2f}")
    ci_viz_dict = cost_by_viz["confidence_intervals_viz"]
    print(f"cell PairWise CI_viz: {ci_viz_dict[('line_charts', 'heatmaps')]}")
    metric_viz = ci_viz_dict[("line_charts", "heatmaps")]['metric']
    ci_viz = ci_viz_dict[("line_charts", "heatmaps")]['ci']
    cells.append(f"{metric_viz:.2f}{ci_viz}")
    print(f"""Effect sizes and confidence intervals for {task}""")
    table_row = str.join(" & ", cells)
    table_row = table_row.replace(")", "]").replace("(", "[")
    print(f"LaTeX table row for task {task} and metric name {metric_name}: & {table_row} \\\\")


def main():
    df = read_data()
    eq_df, cvd_df, t1_df_merged, t2_df_merged, t3_df_merged, t4_df_merged, end_q_df = data_set_creator(df)
    common_users = get_user_intersection(t1_df_merged, t2_df_merged, t3_df_merged, t4_df_merged)
    # removes the members who aborted and have not completed the tasks till the end
    t1_common, t2_common, t3_common, t4_common = common_members_dfs(t1_df_merged, t2_df_merged, t3_df_merged,
                                                                    t4_df_merged, common_users)
    datasets = [t1_common, t2_common, t3_common, t4_common]
    tasks = [task.value for task in Tasks]
    results = {}
    for i, task_number in enumerate(tasks):
        print(f"Starting analysis for task {i+1}")
        stats = task_plotter(datasets[i], DependentVariablePerTask[task_number].value, task_number)
        print_effect_sizes_with_ci_as_table(stats, task_number)
        print(f"Completed analysis for task {i+1}")
        results[f"task_{i+1}"] = stats
    print("Analysis complete.")


if __name__ == '__main__':
    main()
