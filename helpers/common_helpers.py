from typing import Union
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scikit_posthocs import posthoc_dunn
from scipy.stats import shapiro, kstest, normaltest, norm, kruskal, f_oneway, tukey_hsd
from itertools import combinations
import pprint
from config import ROUNDING_DIGITS
from enum import Enum

class Tasks(Enum):
    TASK1 = "task1"
    TASK2 = "task2"
    TASK3 = "task3"
    TASK4 = "task4"

class DependentVariables(Enum):
    HISTORY_LOWEST_ELEC_COST = "history_lowest_elec_cost"
    HISTORY_LOWEST_OVERALL_COST = "history_lowest_overall_cost"
    HISTORY_HIGHEST_OVERALL_COST = "history_highest_overall_cost"

class DependentVariablePerTask(Enum):
    task1 = DependentVariables.HISTORY_LOWEST_ELEC_COST.value
    task2 = DependentVariables.HISTORY_LOWEST_OVERALL_COST.value
    task3 = DependentVariables.HISTORY_LOWEST_OVERALL_COST.value
    task4 = DependentVariables.HISTORY_HIGHEST_OVERALL_COST.value


class VisualizationTypes(Enum):
    LINE_CHARTS = "lc"
    COLORFIELDS = "h" # colorfields in the publication


def is_data_normal(data: [], method="normaltest") -> bool:
    if method == "normaltest":
        stat, p = normaltest(data)
    elif method == "shapiro":
        stat, p = shapiro(data)
    if p < 0.05:
        return False
    else:
        return True


def is_the_mean_different(*samples, method="one_way_anova") -> bool:
    if method == "one_way_anova":
        stat, p = f_oneway(*samples[0])
    else:
        stat, p = kruskal(*samples[0])
    if p < 0.05:
        return True
    else:
        return False

def get_global_effect_size(samples: list, method:str = "one_way_anova") -> dict:
    """
    Calculate the effect size (Cohen's d) for multiple samples.
    :param samples: list of samples (each sample is a list of values)
    :return: effect size (Cohen's d)
    """
    if method == "one_way_anova":
        # stat, p = f_oneway(*samples[0])
        all_data = np.concatenate(samples)
        grand_mean = np.mean(all_data)
        ss_between = sum([len(sample) * (np.mean(sample) - grand_mean) ** 2 for sample in samples])
        ss_within = sum([sum((sample - np.mean(sample)) ** 2) for sample in samples])
        df_between = len(samples) - 1
        df_within = len(all_data) - len(samples)
        # ms_between = ss_between / df_between
        ms_within = ss_within / df_within
        # f_value = ms_between / ms_within
        # eta_squared = ss_between / (ss_between + ss_within)
        omega_squared = (ss_between - (df_between * ms_within)) / (ss_between + ss_within + ms_within)
        effect_size = {"metric": omega_squared, "amplitude": get_cohen_assessment_global_effect_size(omega_squared), "metric_name":"omega_squared", "type":"parametric"}
    else:
        stat, p = kruskal(*samples)
        epsilon_squared = (stat - len(samples) + 1) / (sum([len(sample) for sample in samples]) - len(samples))
        effect_size = {"metric": epsilon_squared, "amplitude": get_cohen_assessment_global_effect_size(epsilon_squared), "metric_name":"epsilon_squared", "type":"non_parametric"}
    return effect_size


def get_cohen_assessment_global_effect_size(cohen_d: float) -> str:
    if cohen_d < 0.01:
        return "small"
    elif 0.01 <= cohen_d < 0.14:
        return "medium"
    else:
        return "large"


def get_cohen_pairwise_effect_size(sample1: list, sample2: list) -> float:
    """
    Calculate the effect size (Cohen's d) between two samples.
    :param sample1: first sample (list of values)
    :param sample2: second sample (list of values)
    :return: effect size (Cohen's d)
    """
    n1 = len(sample1)
    n2 = len(sample2)
    s1 = np.std(sample1, ddof=1)
    s2 = np.std(sample2, ddof=1)
    s_pooled = np.sqrt(((n1 - 1) * s1 ** 2 + (n2 - 1) * s2 ** 2) / (n1 + n2 - 2))
    cohen_d = (np.mean(sample1) - np.mean(sample2)) / s_pooled
    return cohen_d

def get_parametric_pairwise_effect_size(samples: list[list[float]], sample_names: list[str] = None) -> float:
    pairwise_effect_sizes = {}
    if sample_names is None:
        sample_names = [f"sample_{i+1}" for i in range(len(samples))]
    for i in range(len(samples)):
        for j in range(i + 1, len(samples)):
            cohen_d = get_cohen_pairwise_effect_size(samples[i], samples[j])
            if len(samples[i]) < 20 or len(samples[j]) < 20:
                hedges_g = get_hedges_g_pairwise_effect_size(samples[i], samples[j], cohen_d)
                pairwise_effect_sizes[f"{sample_names[i]}_vs_{sample_names[j]}"] = hedges_g
            else:
                pairwise_effect_sizes[f"{sample_names[i]}_vs_{sample_names[j]}"] = cohen_d
    return pairwise_effect_sizes


def get_hedges_g_pairwise_effect_size(sample1: list, sample2: list, cohen_d: float) -> float:
    """
    Calculate the effect size (Hedges' g) between two samples.
    :param sample1: first sample (list of values)
    :param sample2: second sample (list of values)
    :param cohen_d: Cohen's d effect size
    :return: effect size (Hedges' g)
    """
    n1 = len(sample1)
    n2 = len(sample2)
    correction_factor = 1 - (3 / (4 * (n1 + n2) - 9))
    hedges_g = cohen_d * correction_factor
    return hedges_g

def get_non_parametric_pairwise_effect_size(samples: list[list[float]], sample_names: list[str] = None) -> float:
    pairwise_effect_sizes = {}
    if sample_names is None:
        sample_names = [f"sample_{i+1}" for i in range(len(samples))]
    rank_sums = get_rank_sums_per_group(samples)
    total_n = sum([len(sample) for sample in samples])
    for i in range(len(samples)):
        for j in range(i + 1, len(samples)):
            rank_sum1 = rank_sums[f"group_{i+1}"]
            rank_sum2 = rank_sums[f"group_{j+1}"]
            n1 = len(samples[i])
            n2 = len(samples[j])
            dunn_z = get_Dunn_z_score(rank_sum1, rank_sum2, n1, n2)
            r_effect_size = get_rank_based_r_effect_size(dunn_z, total_n)
            pairwise_effect_sizes[f"{sample_names[i]}_vs_{sample_names[j]}"] = r_effect_size
    return pairwise_effect_sizes


def get_Dunn_z_score(rank_sum1: float, rank_sum2: float, n1: int, n2: int) -> float:
    numerator = rank_sum1/n1 - rank_sum2/n2
    denominator = np.sqrt((n1 + n2) * (n1 + n2 + 1) / 12 * (1 / n1 + 1 / n2))
    dunn_z = numerator / denominator
    return dunn_z


def get_rank_sums_per_group(data: list[list[float]], group_names: list[str] = None) -> dict:
    if group_names is None:
        group_names = [f"group_{i+1}" for i in range(len(data))]
    rank_sums = {}
    all_data = np.concatenate(data)
    ranks = pd.Series(all_data).rank().to_list()
    start = 0
    for index, group in enumerate(data):
        n = len(group)
        group_ranks = ranks[start:start + n]
        rank_sums[group_names[index]] = sum(group_ranks)
        start += n
    return rank_sums


def get_rank_based_r_effect_size(dunn_z:float, total_n:float) -> float:
    r = abs(dunn_z) / np.sqrt(total_n)
    return r

def get_r_rank_assessment_global_effect_size(r:float) -> str:
    if r < 0.1:
        return "small"
    elif 0.1 <= r < 0.5:
        return "medium"
    else:
        return "large"

def get_main_stats(
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

def generate_standard_plot_and_stats_for_1d_data(data: Union[pd.Series, list], show_plot=True, xlab="",
                                                 ylab="") -> dict:
    if show_plot:
        plt.figure()
        plt.boxplot(data)
        plt.violinplot(data, widths=0.3)
        plt.scatter(np.ones(len(data)), data)
        plt.scatter(np.linspace(0.5, 1.5, len(data)), data)
        plt.xlim(0.2, 1.7)
        plt.ylabel(ylab)
        plt.xlabel(xlab)
        plt.grid()
        # plt.show()
    minimum = min(data)
    maximum = max(data)
    median = np.median(data)
    average = np.average(data)
    q75, q25 = np.percentile(data, [75, 25])
    iqr = q75 - q25
    n = len(data)
    stdev = np.std(data)
    if isinstance(data, pd.Series):
        data = data.to_list()
    norm_test = np.round(normaltest(data).pvalue, 3)
    shap = np.round(shapiro(data).pvalue, 3)
    kolmogorov = np.round(kstest(data, norm.cdf).pvalue, 3)
    loc = locals()
    stats = dict(
        [(i, np.round(loc[i] * 100) / 100) for i in
         ("minimum", "maximum", "median", "average", "q25", "q75", "iqr", "n", "stdev")])
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(stats)
    normality = dict(
        [(i, np.round(loc[i] * 100) / 100) for i in ("shap", "kolmogorov", "norm_test")])
    pp.pprint(normality)
    return stats

def entry_questionnaire_analyzer(df: pd.DataFrame) -> pd.DataFrame:
    eq_df = df[df["stepName"] == "entry-questionnaire"]
    eq_df.reset_index(inplace=True)
    vals = eq_df["stepValue"].to_list()
    new_vals = []
    for index, value in enumerate(vals):
        val = value.replace("$$", "")
        complete_val = val
        complete_val = complete_val.replace('true', '"true"')
        complete_val = complete_val.replace('false', '"false"')
        complete_val = complete_val.replace('""', '"')
        cv = eval(complete_val)
        entry_dict = cv["entryQuestionnaire"]
        cv.pop("entryQuestionnaire")
        new_vals.append({**entry_dict, **cv})
    vals_df = pd.DataFrame(new_vals)
    eq_final = pd.concat([eq_df.loc[:, ["userID", "sessionID"]], vals_df], axis=1)
    return eq_final


def get_user_intersection(*args: pd.DataFrame) -> list:
    common_users = []
    for index, arg in enumerate(args):
        if index == 0:
            common_users = list(arg["userID"].unique())
            continue
        temp_users = list(arg["userID"].unique())
        new_common_users = [user for user in temp_users if user in common_users]
        common_users = new_common_users
    return common_users

def end_questionnaire_analyzer(df: pd.DataFrame) -> pd.DataFrame:
    eq_df = df[df["stepName"] == "end-questionnaire"]
    eq_df.reset_index(inplace=True)
    vals = eq_df["stepValue"].to_list()
    new_vals = []
    for index, value in enumerate(vals):
        val = value.replace("$$", "")
        complete_val = val.replace('true', '"true"')
        complete_val = complete_val.replace('false', '"false"')
        complete_val = complete_val.replace('\\"', '')
        new_vals.append(eval(complete_val))
    vals_df = pd.DataFrame(new_vals)
    eq_final = pd.concat([eq_df.loc[:, ["userID", "sessionID"]], vals_df], axis=1)
    return eq_final

def get_confidence_intervals_parametric_for_2_groups(n_group1: int, n_group2:int, cohens_d: float = None, hedges_g:float = None, small_sample_adjusted: bool = False) -> dict:
    """Calculate confidence intervals for Cohen's d or Hedge's g using a parametric approach."""
    correction_factor = 1 - (3 / (4 * (n_group1 + n_group2) - 9))
    if cohens_d is None:
        small_sample_adjusted = True
        cohens_d = hedges_g / correction_factor
    se_d = np.sqrt((n_group1 + n_group2) / (n_group1 * n_group2) + (cohens_d ** 2) / (2 * (n_group1 + n_group2)))
    ci_lower = np.round(cohens_d - 1.96 * se_d, ROUNDING_DIGITS)
    ci_upper = np.round(cohens_d + 1.96 * se_d, ROUNDING_DIGITS)
    if not small_sample_adjusted:
        return {"metric": cohens_d,"ci": (ci_lower, ci_upper), "type":"parametric", "metric_name":"cohens_d"}
    else:
        # means we passed hedge's g
        se_g = se_d * correction_factor
        ci_lower = np.round(hedges_g - 1.96 * se_g, ROUNDING_DIGITS)
        ci_upper = np.round(hedges_g + 1.96 * se_g, ROUNDING_DIGITS)
    return {"metric": hedges_g,"ci":(ci_lower, ci_upper), "type":"parametric", "metric_name":"hedges_g"}

def get_confidence_intervals_parametric(samples: list[list[float]], group_names: list[str] = None) -> dict:
    if group_names is None:
        group_names = [f"group_{i+1}" for i in range(len(samples))]
    cis = {}
    for comb in combinations(range(len(samples)), 2):
        i, j = comb
        sample1 = samples[i]
        sample2 = samples[j]
        n1 = len(sample1)
        n2 = len(sample2)
        cohen_d = get_cohen_pairwise_effect_size(sample1, sample2)
        if n1 < 20 or n2 < 20:
            hedges_g = get_hedges_g_pairwise_effect_size(sample1, sample2, cohen_d)
            ci = get_confidence_intervals_parametric_for_2_groups(n1, n2, hedges_g=hedges_g, small_sample_adjusted=True)
            cis[(group_names[i], group_names[j])] = ci
        else:
            ci = get_confidence_intervals_parametric_for_2_groups(n1, n2, cohens_d=cohen_d, small_sample_adjusted=False)
            cis[(group_names[i], group_names[j])] = ci
    return cis

def get_confidence_intervals_non_parametric(samples: list[list[float]], group_names: list[str] = None, n_boot=2000, alpha=0.05) -> dict:
    N = sum(len(sample) for sample in samples)
    if group_names is None:
        group_names = [f"group_{i+1}" for i in range(len(samples))]
    cis = {g:[] for g in list(combinations(group_names, 2))}
    for _ in range(n_boot):
        # Resample within each group
        resampled = [np.random.choice(sample, size=len(sample), replace=True) for sample in samples]
        # Compute rank sums for each group
        rank_sums = get_rank_sums_per_group(resampled, group_names)
        # Compute Dunn z-score and rank-based r for first two groups
        for comb in combinations(rank_sums.keys(), 2):
            group_1, group_2 = comb
            rank_sum1 = rank_sums[group_1]
            rank_sum2 = rank_sums[group_2]
            n1 = len(resampled[int(group_names.index(group_1))])
            n2 = len(resampled[int(group_names.index(group_2))])
            dunn_z = get_Dunn_z_score(rank_sum1, rank_sum2, n1, n2)
            r = dunn_z / np.sqrt(N)
            cis[comb].append(r)
    # Compute CI
    for comb, r_values in cis.items():
        lower = np.round(np.percentile(r_values, 100 * alpha / 2), ROUNDING_DIGITS)
        upper = np.round(np.percentile(r_values, 100 * (1 - alpha / 2)), ROUNDING_DIGITS)
        r_mean = np.round(np.mean(r_values), ROUNDING_DIGITS)
        cis[comb] = {"metric": r_mean, "ci": (lower, upper), "metric_name":"rank_based_r", "type":"non_parametric"}
    return cis


def are_tasks_understood(item: dict, userID: str) -> bool:
    try:
        selected_answer = list(item["checkboxesStatuses"].values()).index('true') + 1
    except ValueError:
        return False
    if item["correctAnswer"] == selected_answer:
        return True
    else:
        return False

def cvd_analyzer(df: pd.DataFrame) -> pd.DataFrame:
    cvd = df[df["stepName"] == "cvd-test"]
    cvd.reset_index(inplace=True)
    vals = cvd["stepValue"].to_list()
    vals_dict = [eval(item.replace("$$", ""))["recordedResults"] for item in vals]
    vals_df = pd.DataFrame(vals_dict)
    cvd_df = pd.concat([cvd.loc[:, ["userID", "sessionID"]], vals_df], axis=1)
    return cvd_df

def common_members_dfs(t0_df_merged, t1_df_merged, t2_df_merged, t3_df_merged, common_users):
    t0_common = t0_df_merged[t0_df_merged["userID"].isin(common_users)]
    t1_common = t1_df_merged[t1_df_merged["userID"].isin(common_users)]
    t2_common = t2_df_merged[t2_df_merged["userID"].isin(common_users)]
    t3_common = t3_df_merged[t3_df_merged["userID"].isin(common_users)]
    return t0_common, t1_common, t2_common, t3_common