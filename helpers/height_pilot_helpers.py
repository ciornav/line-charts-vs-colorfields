import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple
from helpers.common_helpers import is_data_normal, generate_standard_plot_and_stats_for_1d_data, get_main_stats


HEIGHTS = ["20px", "30px", "40px"]


def get_1D_statistics_for_each_height(colorfield_df:pd.DataFrame, line_chart_df:pd.DataFrame, dependent_variable:str, task_number:int) -> Tuple[list, list]:
    lc_stats = []
    h_stats = []
    for height in line_chart_df["scent_height"].unique():
        metric_subset_lc = line_chart_df[line_chart_df["scent_height"] == height][dependent_variable].reset_index(
            drop=True)
        print(
            f"===============generating cost statistics task {task_number} for line_charts and height: {height} ==================")
        d_lc = generate_standard_plot_and_stats_for_1d_data(metric_subset_lc, show_plot=False)
        d_lc["name"] = height
        lc_stats.append(d_lc)
        metric_subset_cf = colorfield_df[colorfield_df["scent_height"] == height][dependent_variable].reset_index(
            drop=True)
        print(
            f"===============generating cost statistics for task {task_number} colorfields and height: {height} ==================")
        d = generate_standard_plot_and_stats_for_1d_data(metric_subset_cf, show_plot=False)
        d["name"] = height
        h_stats.append(d)
    return lc_stats, h_stats


def stats_means_accuracy_height(df: pd.DataFrame, dependent_variable_name: str, heights: list[str]) -> dict:
    line_chart_df = df[df["visualization"] == "line_chart"]
    heatmap_df = df[df["visualization"] == "heatmap"]
    lc_dfs = [line_chart_df[line_chart_df["scent_height"] == height][dependent_variable_name].to_list() for height in heights]
    h_dfs = [heatmap_df[heatmap_df["scent_height"] == height][dependent_variable_name].to_list() for height in heights]
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
    resp = get_main_stats(lc_dfs, "lc", HEIGHTS, is_parametric=lc_normality)
    payload.update(resp)
    resp = get_main_stats(h_dfs, "h", HEIGHTS, is_parametric=h_normality)
    payload.update(resp)
    return payload


def stats_means_accuracy_viz(df: pd.DataFrame, dependent_variable_name: str) -> dict:
    line_charts = df[df["visualization"] == "line_chart"][dependent_variable_name].to_list()
    heatmaps = df[df["visualization"] == "heatmap"][dependent_variable_name].to_list()
    dfs = [line_charts, heatmaps]
    factor_levels = ["line_charts", "heatmaps"]
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


def stats_means_time_height(df: pd.DataFrame, dependent_variable_name: str = "time_metric", heights:list[str] = None) -> dict:
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
    lc_dfs = [line_chart_df[line_chart_df["scent_height"] == height][dependent_variable_name].to_list() for height in
              heights]
    h_dfs = [heatmap_df[heatmap_df["scent_height"] == height][dependent_variable_name].to_list() for height in heights]
    stats_1d_lc, stats_1d_h = {}, {}
    for index, l in enumerate(lc_dfs):
        print(f"=====statistics for  line charts {HEIGHTS[index]}. Time till completion vs height.========")
        stats_1d_lc[f"lc{HEIGHTS[index]}"] = generate_standard_plot_and_stats_for_1d_data(l, show_plot=False)
        print(f"=====statistics for  colorfields {HEIGHTS[index]}. Time till completion vs height.========")
        stats_1d_h[f"h{HEIGHTS[index]}"] = generate_standard_plot_and_stats_for_1d_data(h_dfs[index], show_plot=False)
    payload = {}
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
    payload["are_h_populations_normal"] = h_normality
    resp_lc = get_main_stats(lc_dfs, "lc", HEIGHTS, is_parametric=lc_normality)
    payload.update(resp_lc)
    resp_h = get_main_stats(h_dfs, "h", HEIGHTS, is_parametric=h_normality)
    payload.update(resp_h)
    line_chart_df.boxplot(column="time_metric", by=["scent_height"])
    plt.title("line_chart")
    plt.ylabel("time to solve task [s]")
    # plt.show()
    heatmap_df.boxplot(column="time_metric", by=["scent_height"])
    plt.title("heatmap")
    plt.ylabel("time to solve task [s]")
    # plt.show()
    return payload


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
    factor_levels = ["line_charts", "heatmaps"]
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


def concater(df1: pd.DataFrame, df2: pd.DataFrame):
    common_cols = [col for col in df1.columns if col in df2.columns]
    concated_df = pd.concat([df1[common_cols], df2[common_cols]], ignore_index=True, axis=0)
    return concated_df


def boxplots(colorfield_df:pd.DataFrame, line_chart_df:pd.DataFrame, dependent_variable_name:str, task_data_df:pd.DataFrame):
    line_chart_df.boxplot(column=dependent_variable_name, by=["scent_height"])
    plt.title("line_chart")
    plt.ylabel(f"{dependent_variable_name} [-]")
    colorfield_df.boxplot(column=dependent_variable_name, by=["scent_height"])
    plt.title("colorfields")
    plt.ylabel(f"{dependent_variable_name} [-]")
    task_data_df.boxplot(column=dependent_variable_name, by=["visualization"])
    plt.title("visualization")
    plt.ylabel(f"{dependent_variable_name} [-]")
    plt.show()