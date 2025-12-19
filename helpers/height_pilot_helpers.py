import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple
from scipy.stats import tukey_hsd
from scikit_posthocs import posthoc_dunn
from helpers.common_helpers import is_data_normal, is_the_mean_different, generate_standard_plot_and_stats_for_1d_data, \
    get_global_effect_size, get_non_parametric_pairwise_effect_size, get_parametric_pairwise_effect_size, \
    get_confidence_intervals_non_parametric, get_confidence_intervals_parametric


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


def _get_main_stats(
        groups: list[list[float]],
        factor_name: str,
        factor_levels: list[str],
        is_parametric: bool = True,
) -> dict:
    payload = {}
    if is_parametric:
        are_means_different = is_the_mean_different(groups)
        payload[f"effect_size_{factor_name}"] = get_global_effect_size(groups, method="one_way_anova")
        res = tukey_hsd(*groups)
        payload[f"tukey_hsd_{factor_name}"] = res
        pairwise_effects = get_parametric_pairwise_effect_size(groups, factor_levels)
        ci = get_confidence_intervals_parametric(groups, factor_levels)

    else:
        are_means_different = is_the_mean_different(groups, method="kruskal")
        payload[f"effect_size_{factor_name}"] = get_global_effect_size(groups, method="kruskal")
        res = posthoc_dunn(groups, p_adjust='holm')
        payload["dunn_lc"] = res
        pairwise_effects = get_non_parametric_pairwise_effect_size(groups, factor_levels)
        ci = get_confidence_intervals_non_parametric(groups, factor_levels)
    payload[f"pairwise_effects_{factor_name}"] = {"pairwise_effects": pairwise_effects, "type": "parametric" if is_parametric else "non_parametric"}
    payload[f"confidence_intervals_{factor_name}"] = ci
    payload[f"are_{factor_name}_means_different"] = are_means_different
    return payload

def stats_means_accuracy_height(df: pd.DataFrame, dependent_variable_name: str) -> dict:
    line_chart_df = df[df["visualization"] == "line_chart"]
    heatmap_df = df[df["visualization"] == "heatmap"]
    lc20px = line_chart_df[line_chart_df["scent_height"] == "20px"][dependent_variable_name].to_list()
    lc30px = line_chart_df[line_chart_df["scent_height"] == "30px"][dependent_variable_name].to_list()
    lc40px = line_chart_df[line_chart_df["scent_height"] == "40px"][dependent_variable_name].to_list()
    h20px = heatmap_df[heatmap_df["scent_height"] == "20px"][dependent_variable_name].to_list()
    h30px = heatmap_df[heatmap_df["scent_height"] == "30px"][dependent_variable_name].to_list()
    h40px = heatmap_df[heatmap_df["scent_height"] == "40px"][dependent_variable_name].to_list()
    lc_dfs = [lc20px, lc30px, lc40px]
    h_dfs = [h20px, h30px, h40px]
    factor_levels = ["20px", "30px", "40px"]
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
        resp = _get_main_stats(lc_dfs, "lc", factor_levels, is_parametric=True)
        payload.update(resp)
    else:
        resp = _get_main_stats(lc_dfs, "lc", factor_levels, is_parametric=False)
        payload.update(resp)
    if h_normality:
        resp = _get_main_stats(h_dfs, "h", factor_levels, is_parametric=True)
        payload.update(resp)
    else:
        resp = _get_main_stats(h_dfs, "h", factor_levels, is_parametric=False)
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
    if normality:
        are_means_different = is_the_mean_different(dfs)
        res_lc = tukey_hsd(*dfs)
        payload["tukey_hsd_visualization"] = res_lc
        pairwise_effects = get_parametric_pairwise_effect_size(dfs, factor_levels)
        payload["pairwise_effects_visualization"] = {"pairwise_effects": pairwise_effects, "type": "parametric"}
        ci = get_confidence_intervals_parametric(dfs, factor_levels)
        payload["confidence_intervals_visualization"] = ci
    else:
        are_means_different = is_the_mean_different(dfs, method="kruskal")
        res_lc = posthoc_dunn(dfs, p_adjust='holm')
        payload["dunn_visualization"] = res_lc
        pairwise_effects = get_non_parametric_pairwise_effect_size(dfs, factor_levels)
        payload["pairwise_effects_visualization"] = {"pairwise_effects": pairwise_effects, "type": "non_parametric"}
        confidence_intervals = get_confidence_intervals_non_parametric(dfs, factor_levels)
        payload["confidence_intervals_visualization"] = confidence_intervals
    payload["are_visualization_means_different"] = are_means_different
    payload["are_means_different"] = are_means_different
    return payload


def stats_means_time_height(df: pd.DataFrame, dependent_variable_name: str = "time_metric") -> dict:
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
    lc20px = line_chart_df[line_chart_df["scent_height"] == "20px"][dependent_variable_name].to_list()
    lc30px = line_chart_df[line_chart_df["scent_height"] == "30px"][dependent_variable_name].to_list()
    lc40px = line_chart_df[line_chart_df["scent_height"] == "40px"][dependent_variable_name].to_list()
    h20px = heatmap_df[heatmap_df["scent_height"] == "20px"][dependent_variable_name].to_list()
    h30px = heatmap_df[heatmap_df["scent_height"] == "30px"][dependent_variable_name].to_list()
    h40px = heatmap_df[heatmap_df["scent_height"] == "40px"][dependent_variable_name].to_list()
    lc_dfs = [lc20px, lc30px, lc40px]
    factor_levels = ["20px", "30px", "40px"]
    stats_1d_lc, stats_1d_h = {}, {}
    for index, l in enumerate(lc_dfs):
        if index == 0:
            print("=====statistics for  line charts 20 px. Time till completion vs height.========")
            stats_1d_lc["lc20px"] = generate_standard_plot_and_stats_for_1d_data(l, show_plot=False)
        elif index == 1:
            print("=====statistics for  line charts 30 px. Time till completion vs height.========")
            stats_1d_lc["lc30px"] = generate_standard_plot_and_stats_for_1d_data(l, show_plot=False)
        else:
            print("=====statistics for  line charts 40 px. Time till completion vs height.========")
            stats_1d_lc["lc40px"] = generate_standard_plot_and_stats_for_1d_data(l, show_plot=False)
    h_dfs = [h20px, h30px, h40px]
    for index, h in enumerate(h_dfs):
        if index == 0:
            print("=====statistics for  colorfields 20 px. Time till completion vs height.========")
            stats_1d_h["h20px"] = generate_standard_plot_and_stats_for_1d_data(h, show_plot=False)
        elif index == 1:
            print("=====statistics for  colorfields 30 px. Time till completion vs height.========")
            stats_1d_h["h30px"] = generate_standard_plot_and_stats_for_1d_data(h, show_plot=False)
        else:
            print("=====statistics for  colorfields 40 px. Time till completion vs height.========")
            stats_1d_h["h40px"] = generate_standard_plot_and_stats_for_1d_data(h, show_plot=False)
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
    if lc_normality:
        are_lc_means_different = is_the_mean_different(lc_dfs)
        payload["effect_size_lc"] = get_global_effect_size(lc_dfs, method="one_way_anova")
        res_lc = tukey_hsd(*lc_dfs)
        payload["tukey_hsd_lc"] = res_lc
        pairwise_effects = get_parametric_pairwise_effect_size(lc_dfs, factor_levels)
        payload["pairwise_effects_lc"] = {"pairwise_effects": pairwise_effects, "type": "parametric"}
        ci = get_confidence_intervals_parametric(lc_dfs, factor_levels)
        payload["confidence_intervals_lc"] = ci
    else:
        are_lc_means_different = is_the_mean_different(lc_dfs, method="kruskal")
        payload["effect_size_lc"] = get_global_effect_size(lc_dfs, method="kruskal")
        res_lc = posthoc_dunn(lc_dfs, p_adjust='holm')
        payload["dunn_lc"] = res_lc
        pairwise_effects = get_non_parametric_pairwise_effect_size(lc_dfs, factor_levels)
        payload["pairwise_effects_lc"] = {"pairwise_effects": pairwise_effects, "type": "non_parametric"}
        confidence_intervals = get_confidence_intervals_non_parametric(lc_dfs, factor_levels)
        payload["confidence_intervals_lc"] = confidence_intervals
    payload["are_lc_means_different"] = are_lc_means_different
    if h_normality:
        are_h_means_different = is_the_mean_different(h_dfs)
        payload["effect_size_h"] = get_global_effect_size(h_dfs, method="one_way_anova")
        res_h = tukey_hsd(*h_dfs)
        payload["tukey_hsd_h"] = res_h
        pairwise_effects = get_parametric_pairwise_effect_size(h_dfs, factor_levels)
        payload["pairwise_effects_h"] = {"pairwise_effects": pairwise_effects, "type": "parametric"}
        ci = get_confidence_intervals_non_parametric(h_dfs, factor_levels)
        payload["confidence_intervals_h"] = ci
    else:
        are_h_means_different = is_the_mean_different(h_dfs, method="kruskal")
        payload["effect_size_h"] = get_global_effect_size(h_dfs, method="kruskal")
        res_h = posthoc_dunn(h_dfs, p_adjust='holm')
        payload["dunn_h"] = res_h
        pairwise_effects = get_non_parametric_pairwise_effect_size(h_dfs, factor_levels)
        payload["pairwise_effects_h"] = {"pairwise_effects": pairwise_effects, "type": "non_parametric"}
        confidence_intervals = get_confidence_intervals_non_parametric(lc_dfs, factor_levels)
        payload["confidence_intervals_h"] = confidence_intervals
    payload["are_h_means_different"] = are_h_means_different
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
    if normality:
        are_lc_means_different = is_the_mean_different(dfs)
        res_lc = tukey_hsd(*dfs)
        payload["tukey_hsd_visualization"] = res_lc
        pairwise_effects = get_parametric_pairwise_effect_size(dfs, factor_levels)
        payload["pairwise_effects_visualization"] = {"pairwise_effects": pairwise_effects, "type": "parametric"}
        ci = get_confidence_intervals_parametric(dfs, factor_levels)
        payload["confidence_intervals_visualization"] = ci
    else:
        are_lc_means_different = is_the_mean_different(dfs, method="kruskal")
        res_lc = posthoc_dunn(dfs, p_adjust='holm')
        payload["dunn_visualization"] = res_lc
        pairwise_effects = get_non_parametric_pairwise_effect_size(dfs, factor_levels)
        payload["pairwise_effects_visualization"] = {"pairwise_effects": pairwise_effects, "type": "non_parametric"}
        confidence_intervals = get_confidence_intervals_non_parametric(dfs, factor_levels)
        payload["confidence_intervals_visualization"] = confidence_intervals
    payload["are_means_different"] = are_lc_means_different
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