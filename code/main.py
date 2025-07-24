#!/usr/bin/env python
# coding: utf-8

import qsdsan as qs
import optuna
import numpy as np
from qsdsan import sanunits as su, processes as pc, WasteStream, System
from qsdsan.utils import time_printer, load_data, get_SRT
import warnings
import pandas as pd
import plotly
import plotly.io as pio
from optuna.visualization import plot_contour
from optuna.visualization import plot_edf
from optuna.visualization import plot_intermediate_values
from optuna.visualization import plot_optimization_history
from optuna.visualization import plot_parallel_coordinate
from optuna.visualization import plot_param_importances
from optuna.visualization import plot_slice
import matplotlib
matplotlib.use('Agg')
from datetime import datetime
import os
import time
import optuna
warnings.filterwarnings('ignore')
import psycopg2

def Adjusting_influent_concentration(first_inflow_value, input_COD_value, input_TN_value, input_TP_value):
    adjusting_COD_value = input_COD_value
    adjusting_TN_value = input_TN_value
    adjusting_TP_value = input_TP_value

    def simulate_influent(first_inflow_value, adjusting_COD_value, adjusting_TN_value, adjusting_TP_value):
        Q_inf = first_inflow_value
        Temp = 273.15 + 20
        influent = WasteStream('influent', T=Temp)
        default_inf_kwargs = {  # default influent composition
            'concentrations': {  # you can set concentration of each component separately.
                'S_I': 0,
                'X_I': 0,
                'S_F': 0,
                'S_A': 0,
                'X_S': adjusting_COD_value,
                'S_NH4': 0,
                'S_N2': 0,
                'S_NO3': adjusting_TN_value,
                'S_PO4': adjusting_TP_value,
                'X_PP': 0,
                'X_PHA': 0,
                'X_H': 0.15,
                'X_AUT': 0,
                'X_PAO': 0,
                'S_ALK': 7 * 12,
            },
            'units': ('m3/d', 'mg/L'),  # ('input total flowrate', 'input concentrations')
        }
        influent.set_flow_by_concentration(Q_inf, **default_inf_kwargs)  # set flowrate and composition of empty influent WasteStream
        simulate_influent_COD = round(influent.COD, 2)
        simulate_influent_TN = round(influent.TN, 2)
        simulate_influent_TP = round(influent.TP, 2)
        return simulate_influent_COD, simulate_influent_TN, simulate_influent_TP
    while True:
        simulate_influent_COD, simulate_influent_TN, simulate_influent_TP= simulate_influent(first_inflow_value, adjusting_COD_value, adjusting_TN_value, adjusting_TP_value)
        if (round(simulate_influent_COD, 3) == round(input_COD_value, 3)
            and round(simulate_influent_TN, 3) == round(input_TN_value, 3)
            and round(simulate_influent_TP, 3) == round(input_TP_value, 3)
        ):
            break
        difference_COD = input_COD_value - simulate_influent_COD
        adjusting_COD_value += difference_COD
        difference_TN = input_TN_value - simulate_influent_TN
        adjusting_TN_value += difference_TN
        difference_TP = input_TP_value - simulate_influent_TP
        adjusting_TP_value += difference_TP
    return adjusting_COD_value, adjusting_TN_value, adjusting_TP_value


def simulate_one(first_inflow_value, external_return_flow_value,
             input_COD_value, output_COD_value, adjusting_COD_value,
             input_TN_value, output_TN_value, adjusting_TN_value,
             input_TP_value, output_TP_value, adjusting_TP_value,
             param_names, param_initial_values, parameter_adjustments, day, name, trial
             ):
    Q_inf = first_inflow_value
    Q_was = 385
    Q_ext = external_return_flow_value
    Temp = 273.15 + 20

    # 创建进水、出水、回流
    influent = WasteStream('influent', T=Temp)
    effluent = WasteStream('effluent', T=Temp)
    int_recycle = WasteStream('internal_recycle', T=Temp)
    ext_recycle = WasteStream('external_recycle', T=Temp)
    wastage = WasteStream('wastage', T=Temp)
    # Set the influent composition
    default_inf_kwargs = {  # default influent composition
        'concentrations': {  # you can set concentration of each component separately.
            'S_I': 0,
            'X_I': 0,
            'S_F': 0,
            'S_A': 0,
            'X_S': adjusting_COD_value,
            'S_NH4': 0,
            'S_N2': 0,
            'S_NO3': adjusting_TN_value,
            'S_PO4': adjusting_TP_value,
            'X_PP': 0,
            'X_PHA': 0,
            'X_H': 0.15,
            'X_AUT': 0,
            'X_PAO': 0,
            'S_ALK': 7 * 12,
        },
        'units': ('m3/d', 'mg/L'),  # ('input total flowrate', 'input concentrations')
    }
    influent.set_flow_by_concentration(Q_inf, **default_inf_kwargs)  # set flowrate and composition of empty influent WasteStream

    V_an = 1000 * 33.33
    V_ae = 1333 * 33.33
    aer1 = pc.DiffusedAeration('aer1', DO_ID='S_O2', KLa=240, DOsat=8.0, V=V_ae)  # Tank 3 & Tank 4
    aer2 = pc.DiffusedAeration('aer2', DO_ID='S_O2', KLa=84, DOsat=8.0, V=V_ae)

    # ASM2d
    asm2d = pc.ASM2d()
    # asm2d.show() # Display the 21 processes in ASM2d
    p2 = asm2d.aero_hydrolysis
    adjusted_params = {}
    for i in range(len(param_names)):
        adjusted_params[param_names[i]] = parameter_adjustments[i]
    p2.set_parameters(**adjusted_params)
    pd.set_option('display.max_columns', None)
    # Tank 1 & Tank 2
    A1 = su.CSTR('A1', ins=[influent, int_recycle, ext_recycle], V_max=V_an,
                 aeration=None, suspended_growth_model=asm2d)
    A2 = su.CSTR('A2', ins=A1 - 0, V_max=V_an,  # ins=A1-0
                 aeration=None, suspended_growth_model=asm2d)
    # Tank 3, Tank 4, Tank 5
    O1 = su.CSTR('O1', ins=A2 - 0, V_max=V_ae, aeration=aer1,
                 DO_ID='S_O2', suspended_growth_model=asm2d)
    O2 = su.CSTR('O2', ins=O1 - 0, V_max=V_ae, aeration=aer1,
                 DO_ID='S_O2', suspended_growth_model=asm2d)
    O3 = su.CSTR('O3', ins=O2 - 0, outs=[int_recycle, 'treated'], split=[0.6, 0.4],
                 V_max=V_ae, aeration=aer2,
                 DO_ID='S_O2', suspended_growth_model=asm2d)
    C1 = su.FlatBottomCircularClarifier('C1', ins=O3 - 1, outs=[effluent, ext_recycle, wastage],
                                        underflow=Q_ext, wastage=Q_was, surface_area=1500,
                                        height=4, N_layer=10, feed_layer=5,
                                        X_threshold=3000, v_max=474, v_max_practical=250,
                                        rh=5.76e-4, rp=2.86e-3, fns=2.28e-3)
    sys = System('example_system', path=(A1, A2, O1, O2, O3, C1), recycle=(int_recycle, ext_recycle))  # Path: Sequence of the Reactor
    # sys.diagram() # System diagram
    df = load_data(r'..\data\raw\Water_Treatment_Data.xlsx', sheet='Sheet1')
    # Create a function to set the initial conditions of the reactor
    def batch_init(sys, df):
        dct = df.to_dict('index')
        u = sys.flowsheet.unit
        for k in [u.A1, u.A2, u.O1, u.O2, u.O3]:
            k.set_init_conc(**dct[k._ID])
        c1s = {k: v for k, v in dct['C1_s'].items() if v > 0}
        c1x = {k: v for k, v in dct['C1_x'].items() if v > 0}
        tss = [v for v in dct['C1_tss'].values() if v > 0]
        u.C1.set_init_solubles(**c1s)
        u.C1.set_init_sludge_solids(**c1x)
        u.C1.set_init_TSS(tss)
    batch_init(sys, df)
    sys.set_dynamic_tracker(influent, effluent, A1, A2, O1, O2, O3, C1)
    sys.set_tolerance(rmol=1e-6)
    biomass_IDs = ('X_H', 'X_PAO', 'X_AUT')
    t = 50
    t_step = 1
    try:
        sys.simulate(
            state_reset_hook='reset_cache',
            t_span=(0, t),
            t_eval=np.linspace(0, t, int(t / t_step) + 1),
            method='BDF',
            # export_state_to=f'sol_{t}d_{method}.xlsx',  # Uncomment to export the simulation results as an Excel file
            atol=1e-6,
            rtol=1e-3
        )
        for s in sys.feeds + sys.products:
            if np.any(np.isnan(s.mass)) or np.any(np.isinf(s.mass)):
                raise ValueError(f"{s.ID} has invalid mass values")
        srt = get_SRT(sys, biomass_IDs)
        # print(f"Estimated SRT assuming at steady state is {round(srt, 2)} days")
    except Exception as e:
        raise

    simulate_COD = effluent.COD
    simulate_TN = effluent.TN
    simulate_TP = effluent.TP
    for result_name, result_value in zip(['simulate_COD', 'simulate_TN', 'simulate_TP'],
                                         [simulate_COD, simulate_TN, simulate_TP]):
        if result_value != result_value or result_value == float('inf') or result_value == float('-inf'):
            return

    COD_percentage_difference = abs((output_COD_value - simulate_COD) / output_COD_value * 100)
    TN_percentage_difference = abs((output_TN_value - simulate_TN) / output_TN_value * 100)
    TP_percentage_difference = abs((output_TP_value-simulate_TP) / output_TP_value * 100)
    print("Percentage_difference_COD", COD_percentage_difference)
    print("Percentage_difference_TN", TN_percentage_difference)
    # print("Percentage_difference_TP", TP_percentage_difference)
    simulate_COD_values.append(simulate_COD)
    simulate_TN_values.append(simulate_TN)
    simulate_TP_values.append(simulate_TP)
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    data = {
        'time': [timestamp],
        'data': day + 1,
        'flow': [Q_inf],
        'simulated_influent_cod': round(influent.COD, 2),
        'influent_cod': input_COD_value,
        'effluent_cod': [output_COD_value],
        'simulated_effluent_cod': [simulate_COD],
        'simulated_influent_tn': round(influent.TN, 2),
        'influent_tn': input_TN_value,
        'effluent_tn': [output_TN_value],
        'simulated_effluent_tn': [simulate_TN],
        'simulated_influent_tp': round(influent.TP, 2),
        'influent_tp': input_TP_value,
        'effluent_tp': [output_TP_value],
        'simulated_effluent_tp': [simulate_TP],
        'cod_percentage_difference': [COD_percentage_difference],
        'tn_percentage_difference': [TN_percentage_difference],
        'tp_percentage_difference': [TP_percentage_difference]
    }
    for i in range(len(param_names)):
        data[f'{param_names[i]}_default_value'] = [param_initial_values[i]]
        data[f'{param_names[i]}_adjusted_value'] = [parameter_adjustments[i]]
    folder_path = fr'..\data\process\{name}\{day}'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    new_df = pd.DataFrame(data)
    file_path = fr'..\data\process\{name}\{day}\{day} {name}_{trial.number}_static_simulation_results.xlsx'
    sheet_name = 'Sheet1'
    if os.path.exists(file_path):
        try:
            existing_df = pd.read_excel(file_path, sheet_name=sheet_name)
            updated_df = pd.concat([existing_df, new_df], ignore_index=True)
        except Exception as e:
            updated_df = new_df
    else:
        updated_df = new_df
    with pd.ExcelWriter(file_path, engine='openpyxl', mode='w') as writer:
        updated_df.to_excel(writer, sheet_name=sheet_name, index=False)
        print(f"The data has been successfully added to {file_path}")
    return COD_percentage_difference, TN_percentage_difference, TP_percentage_difference

def OT(param_names, param_initial_values, day, name_1):
    cmps = pc.create_asm2d_cmps()
    adjusting_COD_value, adjusting_TN_value, adjusting_TP_value = Adjusting_influent_concentration(
        first_inflow_value, input_COD_value, input_TN_value, input_TP_value)

    def objective(trial):
        parameter_adjustments = []
        for i in range(len(param_names)):
            parameter_adjustment = trial.suggest_uniform(
                param_names[i],
                param_initial_values[i] * 0.5,
                param_initial_values[i] * 1.5
            )
            parameter_adjustments.append(parameter_adjustment)

        try:
            COD_percentage_difference, TN_percentage_difference, TP_percentage_difference = simulate_one(
                first_inflow_value, external_return_flow_value,
                input_COD_value, output_COD_value, adjusting_COD_value,
                input_TN_value, output_TN_value, adjusting_TN_value,
                input_TP_value, output_TP_value, adjusting_TP_value,
                param_names, param_initial_values, parameter_adjustments, day, name, trial
            )
            if np.isnan(COD_percentage_difference) or np.isinf(COD_percentage_difference):
                raise optuna.exceptions.TrialPruned()
            if np.isnan(TN_percentage_difference) or np.isinf(TN_percentage_difference):
                raise optuna.exceptions.TrialPruned()
            if name_1 == "COD":
                trial.set_user_attr('TN_percentage_difference', TN_percentage_difference)
                return COD_percentage_difference
            elif name_1 == "TN":
                trial.set_user_attr('COD_percentage_difference', COD_percentage_difference)
                return TN_percentage_difference
        except Exception as e:
            raise optuna.exceptions.TrialPruned()

    # Set the SQLite storage path
    sqlite_path = fr"..\data\process\{day}_{name}.db"
    storage_url = f"sqlite:///{sqlite_path.replace(os.sep, '/')}"
    study = optuna.create_study(
        direction="minimize",
        study_name="parameter_optimization",
        storage=storage_url,
        load_if_exists=True
    )
    study.optimize(
        objective,
        n_trials=70,
        n_jobs=-1,# This is the option for the number of threads , -1 indicates to use the maximum number of threads of the computer is CPU
        catch=(RuntimeError, LookupError, ValueError, AssertionError)
    )

    # Record parameter sensitivity
    importance = optuna.importance.get_param_importances(study)
    importance_df = pd.DataFrame(list(importance.items()), columns=['Parameter', 'Importance'])
    output_path = fr'..\data\process\{name}\{day} {name}sensitivity analysis.xlsx'
    importance_df.to_excel(output_path, index=False)

    # Record the history of parameter adjustments
    graph_cout_1 = plot_optimization_history(study, target=lambda t: t.value)
    graph_cout_2 = plot_parallel_coordinate(study, target=lambda t: t.value)
    filename_1 = fr'..\picture\{name}_day_{day}_parameter_tuning_history.html'
    filename_2 = fr'..\picture\{name}_day_{day}_parameter_relationship_chart.html'
    pio.write_html(graph_cout_2, file=filename_2, auto_open=False)
    pio.write_html(graph_cout_1, file=filename_1, auto_open=False)

    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    best_params = study.best_trial.params
    if name_1 == "COD":
        best_cod = study.best_trial.value
        best_tn = study.best_trial.user_attrs.get('TN_percentage_difference')
    elif name_1 == "TN":
        best_tn = study.best_trial.value
        best_cod = study.best_trial.user_attrs.get('COD_percentage_difference')
    data = {
        'time': [timestamp],
        'days': day + 1,
        'cod_optimal_percentage_difference': [best_cod],
        'tn_optimal_percentage_difference': [best_tn],
    }
    df = pd.DataFrame(data)
    print(df)
    file_path = fr'..\data\result\{name}\{day} {name}_optimal_single_objective_static_simulation_results.xlsx'
    sheet_name = 'Sheet1'
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    try:
        if os.path.exists(file_path):
            with pd.ExcelWriter(file_path, engine='xlsxwriter', mode='w') as writer:
                df.to_excel(writer, sheet_name=sheet_name, index=False)
        else:
            df.to_excel(file_path, sheet_name=sheet_name, index=False)
    except Exception as e:
        print(f"Operation failed. Error message: {e}")

def TT(param_names, param_initial_values, day, name_1):
    cmps = pc.create_asm2d_cmps()
    best_param_values = param_initial_values.copy()

    adjusting_COD_value, adjusting_TN_value, adjusting_TP_value = Adjusting_influent_concentration(first_inflow_value,
                                                                                                   input_COD_value,
                                                                                                   input_TN_value,
                                                                                                   input_TP_value)

    for i in range(len(param_names)):
        print(f"Adjusting parameter: {param_names[i]}")
        best_score_tn = float('inf')  #
        best_score_cod = float('inf')  #
        # Perform 10 parameter adjustments for the i-th parameter, using the original value as the median value, with a fluctuation range of ±50%.
        for k in range(11):
            parameter_adjustments = best_param_values.copy()
            adjustment_factor = 0.5 + (k * 0.1)
            param_adjustment_value = param_initial_values[i] * adjustment_factor
            parameter_adjustments[i] = param_adjustment_value
            class Trial:
                def __init__(self):
                    self.number = 0
            trial = Trial()
            COD_percentage_difference, TN_percentage_difference,TP_percentage_difference = simulate_one(
                first_inflow_value, external_return_flow_value,
                input_COD_value, output_COD_value, adjusting_COD_value,
                input_TN_value, output_TN_value, adjusting_TN_value,
                input_TP_value, output_TP_value, adjusting_TP_value,
                param_names, param_initial_values, parameter_adjustments, day, name, trial
            )
            if name_1 == "COD":
                if COD_percentage_difference < best_score_cod:
                    best_score_tn = TN_percentage_difference
                    best_score_cod = COD_percentage_difference
                    # 记录当前参数的最优调整值
                    best_param_values[i] = param_adjustment_value
            elif name_1 == "TN":
                if TN_percentage_difference < best_score_tn:
                    best_score_tn = TN_percentage_difference
                    best_score_cod = COD_percentage_difference
                    best_param_values[i] = param_adjustment_value
        print(f"{param_names[i]} optimal_value: {best_param_values[i]}, optimal_TN_percentage_difference: {best_score_tn}, optimal_COD_percentage_difference: {best_score_cod}")
    print("Optimal values of all parameters: ", best_param_values)

    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    data = {
        'time': [timestamp],
        'days': day + 1,
        'cod_optimal_percentage_difference': [best_score_cod],
        'tn_optimal_percentage_difference': [best_score_tn],
    }
    df = pd.DataFrame(data)
    print(df)
    file_path = fr'..\data\result\{name}\{day} {name}_optimal_single_objective_static_simulation_results.xlsx'
    sheet_name = 'Sheet1'
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    try:
        with pd.ExcelWriter(file_path, engine='openpyxl', mode='w') as writer:
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    except FileNotFoundError:
        df.to_excel(file_path, sheet_name=sheet_name, index=False)

def MO_OT(param_names, param_initial_values, day, name_1):
    cmps = pc.create_asm2d_cmps()
    adjusting_COD_value, adjusting_TN_value, adjusting_TP_value = Adjusting_influent_concentration(
        first_inflow_value, input_COD_value, input_TN_value, input_TP_value)
    def objective(trial):
        parameter_adjustments = []
        for i in range(len(param_names)):
            parameter_adjustment = trial.suggest_uniform(
                param_names[i],
                param_initial_values[i] * 0.5,
                param_initial_values[i] * 1.5
            )
            parameter_adjustments.append(parameter_adjustment)
        try:
            score_cod, score_tn, score_tp = simulate_one(
                first_inflow_value, external_return_flow_value,
                input_COD_value, output_COD_value, adjusting_COD_value,
                input_TN_value, output_TN_value, adjusting_TN_value,
                input_TP_value, output_TP_value, adjusting_TP_value,
                param_names, param_initial_values, parameter_adjustments, day, name, trial
            )
            if np.isnan(score_cod) or np.isinf(score_cod):
                raise optuna.exceptions.TrialPruned()
            if np.isnan(score_tn) or np.isinf(score_tn):
                raise optuna.exceptions.TrialPruned()
            # if np.isnan(score_tp) or np.isinf(score_tp):
            #     raise optuna.exceptions.TrialPruned()
            return score_cod, score_tn
        except Exception as e:
            raise optuna.exceptions.TrialPruned()

    # Set the SQLite storage path
    sqlite_path = fr"..\data\process\{day}_{name}.db"
    storage_url = f"sqlite:///{sqlite_path.replace(os.sep, '/')}"
    study = optuna.create_study(
        directions=["minimize", "minimize"],
        study_name="parameter_optimization",
        storage=storage_url,
        load_if_exists=True
    )
    study.optimize(
        objective,
        n_trials=70,
        n_jobs=-1,
        catch=(RuntimeError, LookupError, ValueError, AssertionError)
    )

    importance = optuna.importance.get_param_importances(study, target=lambda t: t.values[0])
    importance_df = pd.DataFrame(list(importance.items()), columns=['Parameter', 'Importance'])
    output_path = fr'..\data\process\{name}\{day} {name}_sensitivity_analysis.xlsx'
    importance_df.to_excel(output_path, index=False)

    # Record the history of parameter adjustments
    graph_cout_1 = plot_optimization_history(study, target=lambda t: t.values[0])
    graph_cout_2 = plot_parallel_coordinate(study, target=lambda t: t.values[1])
    filename_1 = fr'..\picture\{name}_day_{day}_parameter_tuning_history.html'
    filename_2 = fr'..\picture\{name}_day_{day}_parameter_relationship_chart.html'
    pio.write_html(graph_cout_2, file=filename_2, auto_open=False)
    pio.write_html(graph_cout_1, file=filename_1, auto_open=False)

    pareto_front = study.best_trials
    def calculate_score(trial):
        cod_value = trial.values[0]
        tn_value = trial.values[1]
        score = cod_value + tn_value
        return score
    scores = [(trial, calculate_score(trial)) for trial in pareto_front]
    best_trial, best_score = min(scores, key=lambda x: x[1])
    best_cod_value = best_trial.values[0]
    best_tn_value = best_trial.values[1]

    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    data = {
        'time': [timestamp],
        'days': day + 1,
        'cod_optimal_percentage_difference': [best_cod_value],
        'tn_optimal_percentage_difference': [best_tn_value],
    }
    df = pd.DataFrame(data)
    print(df)
    file_path = fr'..\data\result\{name}\{day} {name}_optimal_single_objective_static_simulation_results.xlsx'
    sheet_name = 'Sheet1'
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    try:
        if os.path.exists(file_path):
            with pd.ExcelWriter(file_path, engine='xlsxwriter', mode='w') as writer:
                df.to_excel(writer, sheet_name=sheet_name, index=False)
        else:
            df.to_excel(file_path, sheet_name=sheet_name, index=False)
    except Exception as e:
        print(f"Operation failed, error message: {e}")

def MO_TT(param_names, param_initial_values, day, name_1):
    cmps = pc.create_asm2d_cmps()
    best_param_values = param_initial_values.copy()

    adjusting_COD_value, adjusting_TN_value, adjusting_TP_value = Adjusting_influent_concentration(first_inflow_value,
                                                                                                   input_COD_value,
                                                                                                   input_TN_value,
                                                                                                   input_TP_value)

    for i in range(len(param_names)):
        print(f"正在调参: {param_names[i]}")
        best_score_tn = float('inf')
        best_score_cod = float('inf')
        for k in range(11):
            parameter_adjustments = best_param_values.copy()
            adjustment_factor = 0.5 + (k * 0.1)
            param_adjustment_value = param_initial_values[i] * adjustment_factor
            parameter_adjustments[i] = param_adjustment_value
            class Trial:
                def __init__(self):
                    self.number = 0
            trial = Trial()
            COD_percentage_difference, TN_percentage_difference,TP_percentage_difference = simulate_one(
                first_inflow_value, external_return_flow_value,
                input_COD_value, output_COD_value, adjusting_COD_value,
                input_TN_value, output_TN_value, adjusting_TN_value,
                input_TP_value, output_TP_value, adjusting_TP_value,
                param_names, param_initial_values, parameter_adjustments, day, name, trial
            )
            if TN_percentage_difference +COD_percentage_difference < best_score_tn + best_score_cod:
                best_score_tn = TN_percentage_difference
                best_score_cod = COD_percentage_difference
                best_param_values[i] = param_adjustment_value
        print(
            f"{param_names[i]} optimal_value: {best_param_values[i]}, optimal_TN_percentage_difference: {best_score_tn}, optimal_COD_percentage_difference: {best_score_cod}"
        )
    print("Optimal values of all parameters: ", best_param_values)

    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    data = {
        'time': [timestamp],
        'days': day + 1,
        'cod_optimal_percentage_difference': [best_score_cod],
        'tn_optimal_percentage_difference': [best_score_tn],
    }
    df = pd.DataFrame(data)
    print(df)
    file_path = fr'..\data\result\{name}\{day} {name}_optimal_single_objective_static_simulation_results.xlsx'
    sheet_name = 'Sheet1'
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    try:
        with pd.ExcelWriter(file_path, engine='openpyxl', mode='w') as writer:
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    except FileNotFoundError:
        df.to_excel(file_path, sheet_name=sheet_name, index=False)

def OSA(name_1):
    file_path = fr'..\data\raw\{name_1}_OSA_FP.xlsx'
    df = pd.read_excel(file_path)
    top_parameters = df[['Parameter', 'average_value']].sort_values(by='average_value', ascending=False).head(7)
    print(top_parameters)
    original_params = {
        'f_SI': 0, 'Y_H': 0.625, 'f_XI_H': 0.1, 'Y_PAO': 0.625, 'Y_PO4': 0.4,
        'Y_PHA': 0.2, 'f_XI_PAO': 0.1, 'Y_A': 0.24, 'f_XI_AUT': 0.1, 'K_h': 3,
        'eta_NO3': 0.6, 'eta_fe': 0.4, 'K_O2': 0.2, 'K_NO3': 0.5, 'K_X': 0.1,
        'mu_H': 6, 'q_fe': 3, 'eta_NO3_H': 0.8, 'b_H': 0.4, 'K_O2_H': 0.2,
        'K_F': 4, 'K_fe': 4, 'K_A_H': 4, 'K_NO3_H': 0.5, 'K_NH4_H': 0.05,
        'K_P_H': 0.01, 'K_ALK_H': 1.2, 'q_PHA': 3, 'q_PP': 1.5, 'mu_PAO': 1,
        'eta_NO3_PAO': 0.6, 'b_PAO': 0.2, 'b_PP': 0.2, 'b_PHA': 0.2,
        'K_O2_PAO': 0.2, 'K_NO3_PAO': 0.5, 'K_A_PAO': 4, 'K_NH4_PAO': 0.05,
        'K_PS': 0.2, 'K_P_PAO': 0.01, 'K_ALK_PAO': 1.2, 'K_PP': 0.01,
        'K_MAX': 0.34, 'K_IPP': 0.02, 'K_PHA': 0.01, 'mu_AUT': 1, 'b_AUT': 0.15,
        'K_O2_AUT': 0.5, 'K_NH4_AUT': 1, 'K_ALK_AUT': 6, 'K_P_AUT': 0.01,
        'k_PRE': 1, 'k_RED': 0.6, 'K_ALK_PRE': 6
    }
    params = []
    for param in top_parameters['Parameter']:
        if param in original_params:
            params.append({'name': param, 'initial_value': original_params[param]})
    param_names = []
    param_initial_values = []
    for param in params:
        param_name = param['name']
        param_initial_value = param['initial_value']
        param_names.append(param_name)
        param_initial_values.append(param_initial_value)
    return param_names, param_initial_values

def TSA(name_1):
    for a in range(1, 2):
        file_path = fr"..\data\raw\Traditional sensitivity analysis\{a} Traditional sensitivity analysis.xlsx"
        data = pd.read_excel(file_path)
        selected_columns = data[[f'{name_1}_sensitivity_coefficient', 'parameter', 'default_value']]
        top_seven = selected_columns.nsmallest(7, f'{name_1}_sensitivity_coefficient')
        params = [
            {
                'name': row['parameter'],
                'initial_value': row['default_value']
            }
            for _, row in top_seven.iterrows()
        ]
        for param in params:
            print(param, "\n")

    param_names = []
    param_initial_values = []
    for param in params:
        param_name = param['name']
        param_initial_value = param['initial_value']
        param_names.append(param_name)
        param_initial_values.append(param_initial_value)
    return param_names, param_initial_values

def FP(name_1):
    params = [
        {'name': 'f_SI', 'initial_value': 0},
        {'name': 'Y_H', 'initial_value': 0.625},
        {'name': 'f_XI_H', 'initial_value': 0.1},
        {'name': 'Y_PAO', 'initial_value': 0.625},
        {'name': 'Y_PO4', 'initial_value': 0.4},
        {'name': 'Y_PHA', 'initial_value': 0.2},
        {'name': 'f_XI_PAO', 'initial_value': 0.1},
        {'name': 'Y_A', 'initial_value': 0.24},
        {'name': 'f_XI_AUT', 'initial_value': 0.1},
        {'name': 'K_h', 'initial_value': 3},
        {'name': 'eta_NO3', 'initial_value': 0.6},
        {'name': 'eta_fe', 'initial_value': 0.4},
        {'name': 'K_O2', 'initial_value': 0.2},
        {'name': 'K_NO3', 'initial_value': 0.5},
        {'name': 'K_X', 'initial_value': 0.1},
        {'name': 'mu_H', 'initial_value': 6},
        {'name': 'q_fe', 'initial_value': 3},
        {'name': 'eta_NO3_H', 'initial_value': 0.8},
        {'name': 'b_H', 'initial_value': 0.4},
        {'name': 'K_O2_H', 'initial_value': 0.2},
        {'name': 'K_F', 'initial_value': 4},
        {'name': 'K_fe', 'initial_value': 4},
        {'name': 'K_A_H', 'initial_value': 4},
        {'name': 'K_NO3_H', 'initial_value': 0.5},
        {'name': 'K_NH4_H', 'initial_value': 0.05},
        {'name': 'K_P_H', 'initial_value': 0.01},
        {'name': 'K_ALK_H', 'initial_value': 1.2},
        {'name': 'q_PHA', 'initial_value': 3},
        {'name': 'q_PP', 'initial_value': 1.5},
        {'name': 'mu_PAO', 'initial_value': 1},
        {'name': 'eta_NO3_PAO', 'initial_value': 0.6},
        {'name': 'b_PAO', 'initial_value': 0.2},
        {'name': 'b_PP', 'initial_value': 0.2},
        {'name': 'b_PHA', 'initial_value': 0.2},
        {'name': 'K_O2_PAO', 'initial_value': 0.2},
        {'name': 'K_NO3_PAO', 'initial_value': 0.5},
        {'name': 'K_A_PAO', 'initial_value': 4},
        {'name': 'K_NH4_PAO', 'initial_value': 0.05},
        {'name': 'K_PS', 'initial_value': 0.2},
        {'name': 'K_P_PAO', 'initial_value': 0.01},
        {'name': 'K_ALK_PAO', 'initial_value': 1.2},
        {'name': 'K_PP', 'initial_value': 0.01},
        {'name': 'K_MAX', 'initial_value': 0.34},
        {'name': 'K_IPP', 'initial_value': 0.02},
        {'name': 'K_PHA', 'initial_value': 0.01},
        {'name': 'mu_AUT', 'initial_value': 1},
        {'name': 'b_AUT', 'initial_value': 0.15},
        {'name': 'K_O2_AUT', 'initial_value': 0.5},
        {'name': 'K_NH4_AUT', 'initial_value': 1},
        {'name': 'K_ALK_AUT', 'initial_value': 6},
        {'name': 'K_P_AUT', 'initial_value': 0.01},
        {'name': 'k_PRE', 'initial_value': 1},
        {'name': 'k_RED', 'initial_value': 0.6},
        {'name': 'K_ALK_PRE', 'initial_value': 6}
    ]
    param_names = []
    param_initial_values = []
    for param in params:
        param_name = param['name']
        param_initial_value = param['initial_value']
        param_names.append(param_name)
        param_initial_values.append(param_initial_value)
    return param_names, param_initial_values

file_path = r'..\data\raw\input.xlsx'
df = pd.read_excel(file_path)

inflow_data = df['inflow_flow']
external_return_flow = df['external_return_flow']
input_COD_data = df['inflow_COD']
input_TN_data = df['inflow_TN']
input_TP_data = df['inflow_TP']
output_COD_data = df['effluent_COD']
output_TN_data = df['effluent_TN']
output_TP_data = df['effluent_TP']

for day in range(1, 50):
    first_inflow_value = inflow_data.iloc[day]
    external_return_flow_value = external_return_flow.iloc[day]
    input_COD_value = input_COD_data.iloc[day]
    input_TN_value = input_TN_data.iloc[day]
    input_TP_value = input_TP_data.iloc[day]
    output_COD_value = output_COD_data.iloc[day]
    output_TN_value = output_TN_data.iloc[day]
    output_TP_value = output_TP_data.iloc[day]

    simulate_values = []
    simulate_COD_values = []
    simulate_TN_values = []
    simulate_TP_values = []
    best_params = []

    name_1 = 'TN' # optimization objective
    name_2 = 'FP' # Parameter sensitivity analysis method
    name_3 = 'TT' # Parameter tuning methods
    name = f'{name_1}_{name_2}_{name_3}'

    # Obtain the parameter selection function
    param_function = globals().get(name_2)
    if param_function is None:
        raise ValueError(f"Function '{name_2}' does not exist. Please check if it is defined.")
    # Obtain the function for parameter adjustment execution
    tuning_function = globals().get(name_3)
    if tuning_function is None:
        raise ValueError(f"Function '{name_3}' does not exist. Please check if it is defined.")

    # Execute parameter extraction + parameter tuning
    param_names, param_initial_values = param_function(name_1)
    tuning_function(param_names, param_initial_values, day, name_1)




