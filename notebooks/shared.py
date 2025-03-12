import os
import re
import pandas as pd
import numpy as np

def generate_result_dataframe(global_results, null_times):
        # Function to format bounds
    def format_bounds(bounds):
        lower, central, upper = bounds
        return f"${central}_{{-{central-lower:.2g}}}^{{+{upper-central:.2g}}}$"

    def format_times(times):
        return "$"+str(int(sum(times)))+"$"

    global_results_list = [x for x in global_results.values()]
    results = []
    for global_result in global_results_list:
        pippo = global_result
        name = global_result["null_config"]["name"]
        print(f"Metric: {name}")
        deformation = global_result["deformation"]
        bound = global_result["bound"]
        ndims = global_result["null_config"]["test_config"]["ndims"]
        niter = global_result["null_config"]["test_config"]["niter"]
        nsamples = global_result["null_config"]["test_config"]["batch_size_test"]
        try:
            exclusion_95 = eval(format(global_result["exclusion_list"][1][3], ".5f"))
        except:
            exclusion_95 = "N/A"
        try:
            exclusion_99 = eval(format(global_result["exclusion_list"][2][3], ".5f"))
        except:
            exclusion_99 = "N/A"
        time_elapsed = global_result["time_elapsed"]#"$"+str(global_result["time_elapsed"])+"$"
        #print(exclusion_95)
        #print(round_to_n_significant_digits(exclusion_95, 3))
        results.append([name, deformation, ndims, niter, nsamples, bound, exclusion_95, exclusion_99, time_elapsed])
        #results.append([name, deformation, ndims, niter, nsamples, round_to_n_significant_digits(exclusion_95, 3), round_to_n_significant_digits(exclusion_99, 3), int(time_elapsed)])
    results_df = pd.DataFrame(results, columns=["Statistic", "Deformation", "$N$", "$n=m$", "$n_{\\mathrm{iter}}$", "Bound", "$\\epsilon_{95\\%\\mathrm{CL}}$", "$\\epsilon_{99\\%\\mathrm{CL}}$", "t (s)"])
    results_df = results_df.groupby(["Statistic", "Deformation", "$N$", "$n=m$", "$n_{\\mathrm{iter}}$"]).agg({
        "$\\epsilon_{95\\%\\mathrm{CL}}$": lambda x: sorted(x.tolist()),
        "$\\epsilon_{99\\%\\mathrm{CL}}$": lambda x: sorted(x.tolist()),
        "t (s)": lambda x: x.tolist()
    }).reset_index()
    results_df["95%CL"] = results_df["$\\epsilon_{95\\%\\mathrm{CL}}$"]
    results_df["99%CL"] = results_df["$\\epsilon_{99\\%\\mathrm{CL}}$"]
    results_df["time"] = results_df["t (s)"]
    results_df["$\\epsilon_{95\\%\\mathrm{CL}}$"] = results_df["$\\epsilon_{95\\%\\mathrm{CL}}$"].apply(format_bounds)
    results_df["$\\epsilon_{99\\%\\mathrm{CL}}$"] = results_df["$\\epsilon_{99\\%\\mathrm{CL}}$"].apply(format_bounds)
    results_df["t (s)"] = results_df["t (s)"].apply(format_times)
    null_times_latex = [[i,f"${j}$"] for i,j in null_times]
    times_df = pd.DataFrame(null_times_latex+[["lr", "-"]], columns=["Statistic", "$t^{\\mathrm{null}}$ (s)"])
    return [results_df, times_df]

def get_individual_dfs(results_df, show = True):
    # Create sub-dataframes for each deformation
    results_df_mean = results_df[results_df["Deformation"] == "mean"].copy()
    results_df_cov_diag = results_df[results_df["Deformation"] == "cov_diag"].copy()
    results_df_cov_off_diag = results_df[results_df["Deformation"] == "cov_off_diag"].copy()
    results_df_power_abs_up = results_df[results_df["Deformation"] == "power_abs_up"].copy()
    results_df_power_abs_down = results_df[results_df["Deformation"] == "power_abs_down"].copy()
    results_df_random_normal = results_df[results_df["Deformation"] == "random_normal"].copy()
    results_df_random_uniform = results_df[results_df["Deformation"] == "random_uniform"].copy()
    
    # Define the custom order
    custom_order = ['swd', 'ks', 'sks', 'fgd', 'mmd', 'lr']

    results_df_mean.rename(columns={"$\\epsilon_{95\\%\\mathrm{CL}}$": "$\\epsilon^{\\mu}_{95\\%\\mathrm{CL}}$", "$\\epsilon_{99\\%\\mathrm{CL}}$": "$\\epsilon^{\\mu}_{99\\%\\mathrm{CL}}$", "t (s)": "$t^{\\mu}$ (s)"}, inplace=True)
    results_df_mean.drop(columns=["Deformation","$N$","$n=m$","$n_{\\mathrm{iter}}$", "95%CL", "99%CL", "time"], inplace=True)
    # Convert 'Statistic' column to a categorical type with the specified order
    results_df_mean['Statistic'] = pd.Categorical(results_df_mean['Statistic'], categories=custom_order, ordered=True)
    results_df_mean = results_df_mean.sort_values('Statistic')
        
    results_df_cov_diag.rename(columns={"$\\epsilon_{95\\%\\mathrm{CL}}$": "$\\epsilon^{\\Sigma_{ii}}_{95\\%\\mathrm{CL}}$", "$\\epsilon_{99\\%\\mathrm{CL}}$": "$\\epsilon^{\\Sigma_{ii}}_{99\\%\\mathrm{CL}}$", "t (s)": "$t^{\\Sigma_{ii}}$ (s)"}, inplace=True)
    results_df_cov_diag.drop(columns=["Deformation","$N$","$n=m$","$n_{\\mathrm{iter}}$", "95%CL", "99%CL", "time"], inplace=True)
    results_df_cov_diag['Statistic'] = pd.Categorical(results_df_cov_diag['Statistic'], categories=custom_order, ordered=True)
    results_df_cov_diag = results_df_cov_diag.sort_values('Statistic')
        
    results_df_cov_off_diag.rename(columns={"$\\epsilon_{95\\%\\mathrm{CL}}$": "$\\epsilon^{\\Sigma_{i\\neq j}}_{95\\%\\mathrm{CL}}$", "$\\epsilon_{99\\%\\mathrm{CL}}$": "$\\epsilon^{\\Sigma_{i\\neq j}}_{99\\%\\mathrm{CL}}$", "t (s)": "$t^{\\Sigma_{i\\neq j}}$ (s)"}, inplace=True)
    results_df_cov_off_diag.drop(columns=["Deformation","$N$","$n=m$","$n_{\\mathrm{iter}}$", "95%CL", "99%CL", "time"], inplace=True)
    results_df_cov_off_diag['Statistic'] = pd.Categorical(results_df_cov_off_diag['Statistic'], categories=custom_order, ordered=True)
    results_df_cov_off_diag = results_df_cov_off_diag.sort_values('Statistic')
        
    results_df_power_abs_up.rename(columns={"$\\epsilon_{95\\%\\mathrm{CL}}$": "$\\epsilon^{\\rm{pow}_{+}}_{95\\%\\mathrm{CL}}$", "$\\epsilon_{99\\%\\mathrm{CL}}$": "$\\epsilon^{\\rm{pow}_{+}}_{99\\%\\mathrm{CL}}$", "t (s)": "$t^{\\rm{pow}_{+}}$ (s)"}, inplace=True)
    results_df_power_abs_up.drop(columns=["Deformation","$N$","$n=m$","$n_{\\mathrm{iter}}$", "95%CL", "99%CL", "time"], inplace=True)
    results_df_power_abs_up['Statistic'] = pd.Categorical(results_df_power_abs_up['Statistic'], categories=custom_order, ordered=True)
    results_df_power_abs_up = results_df_power_abs_up.sort_values('Statistic')

    results_df_power_abs_down.rename(columns={"$\\epsilon_{95\\%\\mathrm{CL}}$": "$\\epsilon^{\\rm{pow}_{-}}_{95\\%\\mathrm{CL}}$", "$\\epsilon_{99\\%\\mathrm{CL}}$": "$\\epsilon^{\\rm{pow}_{-}}_{99\\%\\mathrm{CL}}$", "t (s)": "$t^{\\rm{pow}_{-}}$ (s)"}, inplace=True)
    results_df_power_abs_down.drop(columns=["Deformation","$N$","$n=m$","$n_{\\mathrm{iter}}$", "95%CL", "99%CL", "time"], inplace=True)
    results_df_power_abs_down['Statistic'] = pd.Categorical(results_df_power_abs_down['Statistic'], categories=custom_order, ordered=True)
    results_df_power_abs_down = results_df_power_abs_down.sort_values('Statistic')

    results_df_random_normal.rename(columns={"$\\epsilon_{95\\%\\mathrm{CL}}$": "$\\epsilon^{\\mathcal{N}}_{95\\%\\mathrm{CL}}$", "$\\epsilon_{99\\%\\mathrm{CL}}$": "$\\epsilon^{\\mathcal{N}}_{99\\%\\mathrm{CL}}$", "t (s)": "$t^{\\mathcal{N}}$ (s)"}, inplace=True)
    results_df_random_normal.drop(columns=["Deformation","$N$","$n=m$","$n_{\\mathrm{iter}}$", "95%CL", "99%CL", "time"], inplace=True)
    results_df_random_normal['Statistic'] = pd.Categorical(results_df_random_normal['Statistic'], categories=custom_order, ordered=True)
    results_df_random_normal = results_df_random_normal.sort_values('Statistic')

    results_df_random_uniform.rename(columns={"$\\epsilon_{95\\%\\mathrm{CL}}$": "$\\epsilon^{\\mathcal{U}}_{95\\%\\mathrm{CL}}$", "$\\epsilon_{99\\%\\mathrm{CL}}$": "$\\epsilon^{\\mathcal{U}}_{99\\%\\mathrm{CL}}$", "t (s)": "$t^{\\mathcal{U}}$ (s)"}, inplace=True)
    results_df_random_uniform.drop(columns=["Deformation","$N$","$n=m$","$n_{\\mathrm{iter}}$", "95%CL", "99%CL", "time"], inplace=True)
    results_df_random_uniform['Statistic'] = pd.Categorical(results_df_random_uniform['Statistic'], categories=custom_order, ordered=True)
    results_df_random_uniform = results_df_random_uniform.sort_values('Statistic')
    
    if show:
        display(results_df_mean)
        display(results_df_cov_diag)
        display(results_df_cov_off_diag)
        display(results_df_power_abs_up)
        display(results_df_power_abs_down)
        display(results_df_random_normal)
        display(results_df_random_uniform)
    
    return [results_df_mean, results_df_cov_diag, results_df_cov_off_diag, results_df_power_abs_up, results_df_power_abs_down, results_df_random_normal, results_df_random_uniform]

def get_sorted_dfs(results_df, show = True):
    # Create sub-dataframes for each deformation
    results_df_mean = results_df[results_df["Deformation"] == "mean"].copy()
    results_df_cov_diag = results_df[results_df["Deformation"] == "cov_diag"].copy()
    results_df_cov_off_diag = results_df[results_df["Deformation"] == "cov_off_diag"].copy()
    results_df_power_abs_up = results_df[results_df["Deformation"] == "power_abs_up"].copy()
    results_df_power_abs_down = results_df[results_df["Deformation"] == "power_abs_down"].copy()
    results_df_random_normal = results_df[results_df["Deformation"] == "random_normal"].copy()
    results_df_random_uniform = results_df[results_df["Deformation"] == "random_uniform"].copy()

    # Sort the dataframes by the 95%CL
    results_df_mean.loc[:, 'sort_key'] = results_df_mean['95%CL'].apply(lambda x: x[1])
    sorted_df_mean = results_df_mean.sort_values(by="sort_key", ascending=True).drop(columns=["sort_key"])
    sorted_df_mean.rename(columns={"$\\epsilon_{95\\%\\mathrm{CL}}$": "$\\epsilon^{\\mu}_{95\\%\\mathrm{CL}}$", "$\\epsilon_{99\\%\\mathrm{CL}}$": "$\\epsilon^{\\mu}_{99\\%\\mathrm{CL}}$", "t (s)": "$t^{\\mu}$ (s)"}, inplace=True)
    if show:
        display(sorted_df_mean)
    sorted_df_mean.drop(columns=["Deformation","$N$","$n=m$","$n_{\\mathrm{iter}}$", "95%CL", "99%CL", "time"], inplace=True)
    if show:
        display(sorted_df_mean)
        
    results_df_cov_diag["sort_key"] = results_df_cov_diag["95%CL"].apply(lambda x: x[1])
    sorted_df_cov_diag = results_df_cov_diag.sort_values(by="sort_key", ascending=True).drop(columns=["sort_key"])
    sorted_df_cov_diag.rename(columns={"$\\epsilon_{95\\%\\mathrm{CL}}$": "$\\epsilon^{\\Sigma_{ii}}_{95\\%\\mathrm{CL}}$", "$\\epsilon_{99\\%\\mathrm{CL}}$": "$\\epsilon^{\\Sigma_{ii}}_{99\\%\\mathrm{CL}}$", "t (s)": "$t^{\\Sigma_{ii}}$ (s)"}, inplace=True)
    if show:
        display(sorted_df_cov_diag)
    sorted_df_cov_diag.drop(columns=["Deformation","$N$","$n=m$","$n_{\\mathrm{iter}}$", "95%CL", "99%CL", "time"], inplace=True)
    if show:
        display(sorted_df_cov_diag)
        
    results_df_cov_off_diag["sort_key"] = results_df_cov_off_diag["95%CL"].apply(lambda x: x[1])
    sorted_df_cov_off_diag = results_df_cov_off_diag.sort_values(by="sort_key", ascending=True).drop(columns=["sort_key"])
    sorted_df_cov_off_diag.rename(columns={"$\\epsilon_{95\\%\\mathrm{CL}}$": "$\\epsilon^{\\Sigma_{i\\neq j}}_{95\\%\\mathrm{CL}}$", "$\\epsilon_{99\\%\\mathrm{CL}}$": "$\\epsilon^{\\Sigma_{i\\neq j}}_{99\\%\\mathrm{CL}}$", "t (s)": "$t^{\\Sigma_{i\\neq j}}$ (s)"}, inplace=True)
    if show:
        display(sorted_df_cov_off_diag)
    sorted_df_cov_off_diag.drop(columns=["Deformation","$N$","$n=m$","$n_{\\mathrm{iter}}$", "95%CL", "99%CL", "time"], inplace=True)
    if show:
        display(sorted_df_cov_off_diag)
        
    results_df_power_abs_up["sort_key"] = results_df_power_abs_up["95%CL"].apply(lambda x: x[1])
    sorted_df_power_abs_up = results_df_power_abs_up.sort_values(by="sort_key", ascending=True).drop(columns=["sort_key"])
    sorted_df_power_abs_up.rename(columns={"$\\epsilon_{95\\%\\mathrm{CL}}$": "$\\epsilon^{\\rm{pow}_{+}}_{95\\%\\mathrm{CL}}$", "$\\epsilon_{99\\%\\mathrm{CL}}$": "$\\epsilon^{\\rm{pow}_{+}}_{99\\%\\mathrm{CL}}$", "t (s)": "$t^{\\rm{pow}_{+}}$ (s)"}, inplace=True)
    if show:
        display(sorted_df_power_abs_up)
    sorted_df_power_abs_up.drop(columns=["Deformation","$N$","$n=m$","$n_{\\mathrm{iter}}$", "95%CL", "99%CL", "time"], inplace=True)
    if show:
        display(sorted_df_power_abs_up)

    results_df_power_abs_down["sort_key"] = results_df_power_abs_down["95%CL"].apply(lambda x: x[1])
    sorted_df_power_abs_down = results_df_power_abs_down.sort_values(by="sort_key", ascending=True).drop(columns=["sort_key"])
    sorted_df_power_abs_down.rename(columns={"$\\epsilon_{95\\%\\mathrm{CL}}$": "$\\epsilon^{\\rm{pow}_{-}}_{95\\%\\mathrm{CL}}$", "$\\epsilon_{99\\%\\mathrm{CL}}$": "$\\epsilon^{\\rm{pow}_{-}}_{99\\%\\mathrm{CL}}$", "t (s)": "$t^{\\rm{pow}_{-}}$ (s)"}, inplace=True)
    if show:
        display(sorted_df_power_abs_down)
    sorted_df_power_abs_down.drop(columns=["Deformation","$N$","$n=m$","$n_{\\mathrm{iter}}$", "95%CL", "99%CL", "time"], inplace=True)
    if show:
        display(sorted_df_power_abs_down)

    results_df_random_normal["sort_key"] = results_df_random_normal["95%CL"].apply(lambda x: x[1])
    sorted_df_random_normal = results_df_random_normal.sort_values(by="sort_key", ascending=True).drop(columns=["sort_key"])
    sorted_df_random_normal.rename(columns={"$\\epsilon_{95\\%\\mathrm{CL}}$": "$\\epsilon^{\\mathcal{N}}_{95\\%\\mathrm{CL}}$", "$\\epsilon_{99\\%\\mathrm{CL}}$": "$\\epsilon^{\\mathcal{N}}_{99\\%\\mathrm{CL}}$", "t (s)": "$t^{\\mathcal{N}}$ (s)"}, inplace=True)
    if show:
        display(sorted_df_random_normal)
    sorted_df_random_normal.drop(columns=["Deformation","$N$","$n=m$","$n_{\\mathrm{iter}}$", "95%CL", "99%CL", "time"], inplace=True)
    if show:
        display(sorted_df_random_normal)

    results_df_random_uniform["sort_key"] = results_df_random_uniform["95%CL"].apply(lambda x: x[1])
    sorted_df_random_uniform = results_df_random_uniform.sort_values(by="sort_key", ascending=True).drop(columns=["sort_key"])
    sorted_df_random_uniform.rename(columns={"$\\epsilon_{95\\%\\mathrm{CL}}$": "$\\epsilon^{\\mathcal{U}}_{95\\%\\mathrm{CL}}$", "$\\epsilon_{99\\%\\mathrm{CL}}$": "$\\epsilon^{\\mathcal{U}}_{99\\%\\mathrm{CL}}$", "t (s)": "$t^{\\mathcal{U}}$ (s)"}, inplace=True)
    if show:
        display(sorted_df_random_uniform)
    sorted_df_random_uniform.drop(columns=["Deformation","$N$","$n=m$","$n_{\\mathrm{iter}}$", "95%CL", "99%CL", "time"], inplace=True)
    if show:
        display(sorted_df_random_uniform)
    
    return [sorted_df_mean, sorted_df_cov_diag, sorted_df_cov_off_diag, sorted_df_power_abs_up, sorted_df_power_abs_down, sorted_df_random_normal, sorted_df_random_uniform]
    

# Function to wrap the specific string with \mathbf{}
def bold_specific_string(cell, specific_string):
    if isinstance(cell, str) and specific_string in cell:
        # Use regex to find the exact match and replace it
        return cell.replace(specific_string, f"${{\\mathbf{{{specific_string[1:-1]}}}}}$")
    return cell

def convert_latex_to_float(string):
    try:
        tmp = string.split("$")[1]
        tmp = tmp.split("_")[0]
        tmp = eval(tmp)
    except:
        tmp = float('inf') if string == '' else eval(string)
    return tmp

def generate_result_latex_wide(results_df, times_df, title):
    # Table with 4 sub-tables
    [results_df_mean, results_df_cov_diag, results_df_cov_off_diag, results_df_power_abs_up, results_df_power_abs_down, results_df_random_normal, results_df_random_uniform] = get_individual_dfs(results_df, show=False)
    # Replace values in the dataframes
    column_replacements = {
        "lr":  "$t_{\\mathrm{LLR}}$",
        "ks":  "$t_{\\overline{\\mathrm{KS}}}$",
        "sks": "$t_{\\mathrm{SKS}}$",
        "swd": "$t_{\\mathrm{SW}}$",
        "fgd": "$t_{\\mathrm{FGD}}$",
        "mmd": "$t_{\\mathrm{MMD}}$",
    }
    
    def replace_columns(df):
        return df.replace(column_replacements)

    # Replace columns in all dataframes
    results_df_mean = replace_columns(results_df_mean)
    results_df_cov_diag = replace_columns(results_df_cov_diag)
    results_df_cov_off_diag = replace_columns(results_df_cov_off_diag)
    results_df_power_abs_up = replace_columns(results_df_power_abs_up)
    results_df_power_abs_down = replace_columns(results_df_power_abs_down)
    results_df_random_normal = replace_columns(results_df_random_normal)
    results_df_random_uniform = replace_columns(results_df_random_uniform)
    times_df = replace_columns(times_df)
    
    result_table_1 = results_df_mean.merge(results_df_cov_diag, on='Statistic', how='outer')
    result_table_1 = result_table_1.merge(results_df_cov_off_diag, on='Statistic', how='outer')
    result_table_1 = result_table_1.merge(results_df_power_abs_up, on='Statistic', how='outer')
    tmp = pd.DataFrame(result_table_1)
    tmp = tmp[tmp['Statistic'] != '$t_{\mathrm{LLR}}$']
    for k in tmp.keys()[1:]:
        tmp.loc[:, 'sort_key'] = tmp[k].apply(convert_latex_to_float)
        tmp.sort_values(by='sort_key', ascending=True, inplace=True)
        bold = list(tmp[k])[0]
        result_table_1 = result_table_1.applymap(lambda x: bold_specific_string(x, bold))
        
    # Table with last 4 columns
    result_table_2 = results_df_power_abs_down.merge(results_df_random_normal, on='Statistic', how='outer')
    result_table_2 = result_table_2.merge(results_df_random_uniform, on='Statistic', how='outer')
    result_table_2 = result_table_2.merge(times_df, on='Statistic', how='outer')
    tmp = pd.DataFrame(result_table_2)
    tmp = tmp[tmp['Statistic'] != '$t_{\mathrm{LLR}}$']
    for k in tmp.keys()[1:]:
        tmp.loc[:, 'sort_key'] = tmp[k].apply(convert_latex_to_float)
        tmp.sort_values(by='sort_key', ascending=True, inplace=True)
        bold = list(tmp[k])[0]
        result_table_2 = result_table_2.applymap(lambda x: bold_specific_string(x, bold))

    #display(result_table)
    # Make Latex Table (first part)
    string = result_table_1.to_latex(index=False, column_format='l|llr|llr|llr|llr', longtable=False, float_format="%.2g")
    # Legend row
    string = string.replace("\\toprule\nStatistic",
        "\\toprule\n\\multicolumn{1}{c}{} \
        & \\multicolumn{3}{c}{$\\mu$-deformation} \
        & \\multicolumn{3}{c}{$\\Sigma_{ii}$-deformation} \
        & \\multicolumn{3}{c}{$\\Sigma_{i\\neq j}$-deformation} \
        & \\multicolumn{3}{c}{$\\rm{pow}_{+}$-deformation} \\\\\nStatistic")
    # Fix sub-columns names
    string = string.replace("^{\\mu}", "")
    string = string.replace("^{\\Sigma_{ii}}", "")
    string = string.replace("^{\\Sigma_{i\\neq j}}", "")
    string = string.replace("^{\\rm{pow}_{+}}", "")
    # Close first table
    string = string.replace("\\bottomrule\n\\end{tabular}\n","")
    string = string.replace("\\toprule","\\toprule\n\\multicolumn{13}{c}{"+title+"} \\\\")
    string = string.rstrip()
    #display(result_table)
    # Make Latex Table (second part)
    string = string + result_table_2.to_latex(index=False, column_format='l|llr|llr|llr|r', longtable=False, float_format="%.2g")
    # Legend row
    string = string.replace("\\toprule\nStatistic",
        "\\toprule\n\\multicolumn{1}{c}{} \
        & \\multicolumn{3}{c}{$\\rm{pow}_{-}$-deformation} \
        & \\multicolumn{3}{c}{$\\mathcal{N}$-deformation} \
        & \\multicolumn{3}{c}{$\\mathcal{U}$-deformation} \
        & \\multicolumn{3}{c}{} \\\\\nStatistic")
    # Fix sub-columns names
    string = string.replace("^{\\rm{pow}_{-}}", "")
    string = string.replace("^{\\mathcal{N}}", "")
    string = string.replace("^{\\mathcal{U}}", "")
    string = string.replace("\\begin{tabular}{l|llr|llr|llr|r}","")
    string = string.lstrip()
    string = string.replace("\n","\n\t")
    string = string.replace("\t\\end{tabular}","\\end{tabular}")
    string = string.replace("NaN","-")
    return string
    
def generate_result_latex(results_df, times_df, title):
    # Table with 2 sub-tables
    [results_df_mean, results_df_cov_diag, results_df_cov_off_diag, results_df_power_abs_up, results_df_power_abs_down, results_df_random_normal, results_df_random_uniform] = get_individual_dfs(results_df, show=False)
    # Replace values in the dataframes
    column_replacements = {
        "lr":  "$t_{\\mathrm{LLR}}$",
        "ks":  "$t_{\\overline{\\mathrm{KS}}}$",
        "sks": "$t_{\\mathrm{SKS}}$",
        "swd": "$t_{\\mathrm{SW}}$",
        "fgd": "$t_{\\mathrm{FGD}}$",
        "mmd": "$t_{\\mathrm{MMD}}$",
    }

    def remove_empty_lines(text):
        return '\n'.join([line for line in text.split('\n') if line.strip() != ''])

    def replace_columns(df):
        return df.replace(column_replacements)

    # Replace columns in all dataframes
    results_df_mean = replace_columns(results_df_mean)
    results_df_cov_diag = replace_columns(results_df_cov_diag)
    results_df_cov_off_diag = replace_columns(results_df_cov_off_diag)
    results_df_power_abs_up = replace_columns(results_df_power_abs_up)
    results_df_power_abs_down = replace_columns(results_df_power_abs_down)
    results_df_random_normal = replace_columns(results_df_random_normal)
    results_df_random_uniform = replace_columns(results_df_random_uniform)
    times_df = replace_columns(times_df)

    # Merge the dataframes for the panels
    result_table_1 = results_df_mean.merge(results_df_cov_diag, on='Statistic', how='outer')
    tmp = pd.DataFrame(result_table_1)
    tmp = tmp[tmp['Statistic'] != '$t_{\mathrm{LLR}}$']
    for k in tmp.keys()[1:]:
        tmp.loc[:, 'sort_key'] = tmp[k].apply(convert_latex_to_float)
        tmp.sort_values(by='sort_key', ascending=True, inplace=True)
        bold = list(tmp[k])[0]
        result_table_1 = result_table_1.applymap(lambda x: bold_specific_string(x, bold))

    result_table_2 = results_df_cov_off_diag.merge(results_df_power_abs_up, on='Statistic', how='outer')
    tmp = pd.DataFrame(result_table_2)
    tmp = tmp[tmp['Statistic'] != '$t_{\mathrm{LLR}}$']
    for k in tmp.keys()[1:]:
        tmp.loc[:, 'sort_key'] = tmp[k].apply(convert_latex_to_float)
        tmp.sort_values(by='sort_key', ascending=True, inplace=True)
        bold = list(tmp[k])[0]
        result_table_2 = result_table_2.applymap(lambda x: bold_specific_string(x, bold))

    result_table_3 = results_df_power_abs_down.merge(results_df_random_normal, on='Statistic', how='outer')
    tmp = pd.DataFrame(result_table_3)
    tmp = tmp[tmp['Statistic'] != '$t_{\mathrm{LLR}}$']
    for k in tmp.keys()[1:]:
        tmp.loc[:, 'sort_key'] = tmp[k].apply(convert_latex_to_float)
        tmp.sort_values(by='sort_key', ascending=True, inplace=True)
        bold = list(tmp[k])[0]
        result_table_3 = result_table_3.applymap(lambda x: bold_specific_string(x, bold))
        
    result_table_4 = results_df_random_uniform.merge(times_df, on='Statistic', how='outer')
    tmp = pd.DataFrame(result_table_4)
    tmp = tmp[tmp['Statistic'] != '$t_{\mathrm{LLR}}$']
    for k in tmp.keys()[1:]:
        tmp.loc[:, 'sort_key'] = tmp[k].apply(convert_latex_to_float)
        tmp.sort_values(by='sort_key', ascending=True, inplace=True)
        bold = list(tmp[k])[0]
        result_table_4 = result_table_4.applymap(lambda x: bold_specific_string(x, bold))

    # Define a function to create LaTeX table code
    def create_latex_table(df, legend):
        latex_code = df.to_latex(index=False, column_format='l|llr|llr', longtable=False, float_format="%.2g")
        latex_code = latex_code.replace("\\toprule\nStatistic", legend)
        latex_code = latex_code.replace("^{\\mu}", "")
        latex_code = latex_code.replace("^{\\Sigma_{ii}}", "")
        latex_code = latex_code.replace("^{\\Sigma_{i\\neq j}}", "")
        latex_code = latex_code.replace("^{\\rm{pow}_{+}}", "")
        latex_code = latex_code.replace("^{\\rm{pow}_{-}}", "")
        latex_code = latex_code.replace("^{\\mathcal{N}}", "")
        latex_code = latex_code.replace("^{\\mathcal{U}}", "")
        return latex_code

    # Legends for the tables
    legend_1 = "\\toprule\n\\multicolumn{1}{c}{} & \\multicolumn{3}{c}{$\\mu$-deformation} & \\multicolumn{3}{c}{$\\Sigma_{ii}$-deformation} \\\\\nStatistic"
    legend_2 = "\\toprule\n\\multicolumn{1}{c}{} & \\multicolumn{3}{c}{$\\Sigma_{i\\neq j}$-deformation} & \\multicolumn{3}{c}{$\\rm{pow}_{+}$-deformation} \\\\\nStatistic"
    legend_3 = "\\toprule\n\\multicolumn{1}{c}{} & \\multicolumn{3}{c}{$\\rm{pow}_{-}$-deformation} & \\multicolumn{3}{c}{$\\mathcal{N}$-deformation} \\\\\nStatistic"
    legend_4 = "\\toprule\n\\multicolumn{1}{c}{} & \\multicolumn{3}{c}{$\\mathcal{U}$-deformation} & \\multicolumn{3}{c}{Timing} \\\\\nStatistic"

    # Create LaTeX table strings
    latex_table_1 = create_latex_table(result_table_1, legend_1)
    latex_table_2 = create_latex_table(result_table_2, legend_2)
    latex_table_3 = create_latex_table(result_table_3, legend_3)
    latex_table_4 = create_latex_table(result_table_4, legend_4)

    # Combine all LaTeX table strings
    string = f"""
        {latex_table_1}
        {latex_table_2}
        {latex_table_3}
        {latex_table_4}
    """
    string = string.replace("\\begin{tabular}{l|llr|llr}","")
    string = string.replace("\\end{tabular}","")
    string = string.replace("\\toprule","")
    string = string.replace("\\bottomrule","\\toprule")
    string = remove_empty_lines(string)
    string = """\\begin{tabular}{l|llr|llr}
\\toprule
\\multicolumn{7}{c}{"""+title+"""} \\\\
\\toprule
"""+string
    string = string.rstrip("\\toprule")
    string = string+"\\bottomrule\n\\end{tabular}"
    string = string.replace("\n","\n\t")
    string = string.replace("\t\\end{tabular}","\\end{tabular}")
    string = string.replace("NaN","-")
    return string

def save_latex(string, model_dir):
    # Define the filename for the output .tex file
    filename = os.path.join(model_dir,'results_table.tex')

    # Write the LaTeX table code to the .tex file
    with open(filename, 'w') as file:
        file.write(string)

    print(f"LaTeX table code has been written to {filename}")