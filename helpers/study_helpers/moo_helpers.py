import numpy as np


def get_slider_grid(problem) -> dict:
    grid = {}
    for slider in range(len(problem.xu)):  # that's the number of inputs
        values = np.linspace(problem.xl[slider], problem.xu[slider], 21)
        slider_name = "slider_" + str(slider)
        grid[slider_name] = values
    return grid


def approximate_point_to_slider_resolution(pareto_point: list, slider_grid: dict) -> dict:
    approximated_point_on_sliders = []
    indexes_around_pareto = []
    for index, value in enumerate(slider_grid.keys()):
        slider_possible_values = list(slider_grid[value])
        pareto_x_coord = pareto_point[index]
        closeness = 999
        selected_slider_value = ""
        for v in slider_possible_values:
            diff = abs(pareto_x_coord - v)
            if diff < closeness:
                closeness = diff
                selected_slider_value = v
        approximated_point_on_sliders.append(selected_slider_value)
        index_selected_value = slider_possible_values.index(selected_slider_value)
        index_surrounding_pareto = get_rotating_index(index_selected_value, slider_possible_values)
        indexes_around_pareto.append(index_surrounding_pareto)
    grid_around_given_point = create_grid_around_pareto_point(approximated_point_on_sliders, indexes_around_pareto, slider_grid)
    return {"grid_around_given_point": grid_around_given_point, "approximated_point_on_sliders": approximated_point_on_sliders}


def get_rotating_index(index_selected_value : int, slider_possible_values : list, number_of_steps_in_neighborhood: int = 3) -> list:
    indexes_around_pareto = []
    for ind in range(number_of_steps_in_neighborhood):
        inferior_index = index_selected_value - (ind +1)
        superior_index = index_selected_value + (ind +1)
        if inferior_index >= 0 :
            indexes_around_pareto.append(inferior_index)
        if superior_index < len(slider_possible_values):
            indexes_around_pareto.append(superior_index)
    return indexes_around_pareto


def create_grid_around_pareto_point(approximated_pareto: list, indexes_around_pareto: list, slider_grid:dict):
    grid = []
    for ind, x in enumerate(approximated_pareto):
        temp_approximated = [*approximated_pareto]
        for slight_change in indexes_around_pareto[ind]:
            slider_possible_values = list(slider_grid.values())
            temp_approximated[ind] = list(slider_possible_values[ind])[slight_change]
            grid.append([*temp_approximated])
    return grid
