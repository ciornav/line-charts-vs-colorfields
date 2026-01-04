import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple
from helpers.common_helpers import is_data_normal, generate_standard_plot_and_stats_for_1d_data, get_main_stats
from enum import Enum

HEIGHTS = ["20px", "30px", "40px"]

class MetricNames(Enum):
    cohen_d = "cohens_d"
    hedge_g = "hedges_g"
    omega_squared = "omega_squared"
    epsilon_squared = "epsilon_squared"
    rank_based_r = "rank_based_r"

class EffectSizeTypes(Enum):
    small = "small"
    medium = "medium"
    large = "large"

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


def _print_detailed_latex_table_row(
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

def _get_summary_latex_table_row(
        statistics_of_interest: dict,
        task: str,
        metric_name: str,
        factor_levels: list[str],
        global_stats_name: str = "cost_by_height",
        viz_stats_name: str = "cost_by_viz"

) -> None:
    global_stats = statistics_of_interest[global_stats_name]
    cells = []
    for factor_level in factor_levels:
        print(f"cell ES_{factor_level}: {global_stats[f'effect_size_{factor_level}']}")
        effect_value = float(f"{global_stats[f'effect_size_{factor_level}']['metric']:.2f}")
        effect_metric_type = f"{global_stats[f'effect_size_{factor_level}']['metric_name']}"
        cell_content = get_custom_cell(effect_value, MetricNames(effect_metric_type))
        cells.append(cell_content)
    cost_by_viz = statistics_of_interest[viz_stats_name]
    # print(f"cell ES_viz: {cost_by_viz['effect_size_viz']}")
    # global_viz = cost_by_viz["effect_size_viz"]["metric"]
    # cells.append(f"{global_viz:.2f}")
    ci_viz_dict = cost_by_viz["confidence_intervals_viz"]
    print(f"cell PairWise CI_viz: {ci_viz_dict[('line_charts', 'heatmaps')]}")
    metric_viz = float(f"{ci_viz_dict[('line_charts', 'heatmaps')]['metric']:.2f}")
    ci_viz = ci_viz_dict[("line_charts", "heatmaps")]['ci']
    effect_metric_type = ci_viz_dict[("line_charts", "heatmaps")]['metric_name']
    is_statistically_significant = all([0 < ci_viz[0], 0 < ci_viz[1]]) or all([ci_viz[0] < 0, ci_viz[1] < 0])
    effect = f"{metric_viz}[{ci_viz[0]}, {ci_viz[1]}]"
    cell_content = get_custom_cell(effect, MetricNames(effect_metric_type), is_statistically_significant)
    cells.append(cell_content)
    print(f"""Effect sizes and confidence intervals for {task}""")
    table_row = str.join(" & ", cells)
    table_row = table_row.replace(")", "]").replace("(", "[")
    print(f"LaTeX table row for task {task} and metric name {metric_name}: & {table_row} \\\\")
    return table_row

def get_custom_cell(effect_value: float | str, metric_name: MetricNames, is_statistically_significant: bool = False) -> str:
    if isinstance(effect_value, str):
        effect_value_str = effect_value.split("[")[0]
        effect_value_float = float(effect_value_str)
        effect_size = get_interpreted_effect_size(effect_value_float, metric_name)
    else:
        effect_size = get_interpreted_effect_size(effect_value, metric_name)
    metric_name_color = get_color_for_metric_name(metric_name)
    if is_statistically_significant:
        effect_value = str(effect_value) + "*"
    if effect_size == EffectSizeTypes.medium:
        # see command coloredcell in the latex code in the paper
        return "\cellcolor{cyan!15} "  f"{effect_value}" + " & \cellcolor{" +  f"{metric_name_color}" + "}"
    elif effect_size == EffectSizeTypes.large:
        return "\cellcolor{cyan!50} "  f"{effect_value}" + " & \cellcolor{" +  f"{metric_name_color}" + "}"
    else:
        return "\cellcolor{white} "  f"{effect_value}" + " & \cellcolor{" +  f"{metric_name_color}" + "}"

def get_color_for_metric_name(metric_name:MetricNames) -> str:
    if metric_name == MetricNames.epsilon_squared:
        return "blue!45"
    elif metric_name == MetricNames.omega_squared:
        return "green!65"
    elif metric_name == MetricNames.cohen_d:
        return "orange!60"
    elif metric_name == MetricNames.rank_based_r:
        return "magenta!40"
    else:
        raise ValueError(f"Unknown metric name: {metric_name}")

def get_interpreted_effect_size(es_value:float, metric_name:MetricNames) -> EffectSizeTypes:
    if metric_name == MetricNames.omega_squared or metric_name == MetricNames.epsilon_squared:
        es_value = abs(es_value)
        if es_value < 0.13:
            return EffectSizeTypes.small
        elif 0.13 <= es_value < 0.26:
            return EffectSizeTypes.medium
        else:
            return EffectSizeTypes.large
    elif metric_name == MetricNames.rank_based_r:
        abs_value = abs(es_value)
        if abs_value < 0.3:
            return EffectSizeTypes.small
        elif 0.3 <= abs_value < 0.5:
            return EffectSizeTypes.medium
        else:
            return EffectSizeTypes.large
    elif metric_name == MetricNames.cohen_d or metric_name == MetricNames.hedge_g:
        abs_value = abs(es_value)
        if abs_value < 0.5:
            return EffectSizeTypes.small
        elif 0.5 <= abs_value < 0.8:
            return EffectSizeTypes.medium
        else:
            return EffectSizeTypes.large
    else:
        raise ValueError(f"Unknown metric name: {metric_name}")

def create_latex_table(data:dict):
    table_header = """
         \\begin{table}[h!]
         \centering
         \small
         \\resizebox{\columnwidth}{!}{
         \\begin{tabular}{|l|c|c|c|c|c|c|c|c|c|c|c|c|}
         \hline
         \\textbf{}
          & \multicolumn{6}{c|}{\\textbf{Cost}} & \multicolumn{6}{c|}{\\textbf{Completion time}}
          \\\\
         \cline{2-13}
         \\textbf{}
         & \multicolumn{4}{c|}{\\textbf{Global Effect Size}}
         & \multicolumn{2}{c|}{\\textbf{Pairwise ES LC vs C}}
         & \multicolumn{4}{c|}{\\textbf{Global Effect Size}}
         & \multicolumn{2}{c|}{\\textbf{Pairwise ES LC vs C}}
         \\\\
         \cline{2-13}
         \\textbf{}
         & \multicolumn{2}{c|}{\\textbf{LC}}
         & \multicolumn{2}{c|}{\\textbf{C}}
         & \multicolumn{2}{c|}{\\textbf{LC vs C}}
        & \multicolumn{2}{c|}{\\textbf{LC}}
         & \multicolumn{2}{c|}{\\textbf{C}}
         & \multicolumn{2}{c|}{\\textbf{LC vs C}} \\\\
        \hline
    """
    for task in data.keys():
        payload = data[task]
        cost_data = payload["cost_means_stats"]
        cost_table_row_data = _get_summary_latex_table_row(cost_data, task, "cost", factor_levels=["lc", "h"])
        # _print_detailed_latex_table_row(cost_data, "cost_by_height", task, "cost", factor_levels=["lc", "h"])
        table_header += f" {task} & {cost_table_row_data}"
        time_data = payload["time_means_stats"]
        time_table_row_data = _get_summary_latex_table_row(time_data, task, "time", factor_levels=["lc", "h"], global_stats_name="time_to_execute_by_height", viz_stats_name="time_by_viz_lc_or_h")
        # _print_detailed_latex_table_row(time_data, "time_to_execute_by_height", task, "time", factor_levels=["lc", "h"])
        table_header += f" & {time_table_row_data} \\\\  \\hline "

    table_footer = """
     \end{tabular}
     }
     \\vspace{0.15em}
    \\begin{minipage}{\linewidth}
    \centering
    \\tiny
    \setlength{\\tabcolsep}{5pt}
    \\renewcommand{\\arraystretch}{0.9}
    \\begin{tabular}{@{}l l l l l l l l l l l@{}}
    \cellcolor{blue!45}\\rule{0.3em}{0pt}  & $\epsilon^2$ & {} &
    \cellcolor{green!65}\\rule{0.3em}{0pt} & $\omega^2$ & {} &
    \cellcolor{orange!60}\\rule{0.3em}{0pt} & $d$ & {} &
    \cellcolor{magenta!40}\\rule{0.3em}{0pt} & $r$
    \end{tabular}
    \end{minipage}
    
    \\vspace{0.15em}
    \\begin{minipage}{\linewidth}
    \centering
    \\tiny
    \setlength{\\tabcolsep}{5pt}
    \\renewcommand{\\arraystretch}{0.1}
    \\begin{tabular}{@{}l l l l l l l @{}}
    \cellcolor{cyan!15}\\rule{0.4em}{0pt}  & medium effect  & {} & \cellcolor{cyan!50}\\rule{0.4em}{0pt} & large effect & {} &  *   statistically significant
    \end{tabular}
    \end{minipage}
         
     \caption{ Effect sizes for Pilot 1: scent heights. Each efect size value is
                calculated according to the methodology presented in section 4. Scent
                height has no practical effect on user performance. Preliminary data
                in this pilot suggests that line charts allow faster completion times than
                colorfields for Task 1 with a medium effect size.}   
     \label{tab:effect_sizes_tasks_cost_pairwise}
     \end{table}
    """
    full_table = table_header + table_footer
    print("Full LaTeX table:")
    print(full_table)

