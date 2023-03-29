import numpy as np
import argparse
import subprocess
import datetime
import time
import os
import pandas as pd
import glob as glob



def string(boolean):
    if boolean == True:
        return "true"
    else:
        return "false"

def loop(output_dir, params, param_names,
         values = [], depth = 0, **kwargs):
    jobfile = 'parametersearch.sh'
    if kwargs.get('test', False) == True:
        num_exps = 2
    else:
        num_exps = 50
        
    if kwargs.get('analyze_only', False) == True:
        analyze_only = '#'
    else:
        analyze_only = ''
        
    
        
    if depth == len(params):
        cmd_string = ""
        print("values", values)
        for idx, param_name in enumerate(param_names):
            # cmd_string += param_bool + "=" + string(bools[idx]) + " "
            # if bools[idx]: # no need to add value if setaveL is false
            # print('-*-')
            # print('idx' + str(idx))
            # print('test1')
            # print(param_names[idx])
            # print('test2')
            # print('values')
            # print(values[idx])
            # print( param_names[idx] + "---------" + str(values[idx]) + " ")
            cmd_string += param_names[idx] + "=" + str(values[idx]) + " "

        # print(cmd_string)
        # add datetime as name of directory to save file
        saveDirName = f"{output_dir}/results_" #+ datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        for idx, param_name in enumerate(param_names):
            saveDirName += f"{param_name}={str(values[idx])}"

        # Edit sbatch file
        with open(jobfile, "r+") as f:
            lines = f.readlines()

        print(lines)
        with open(jobfile, "w") as f:
            lines[-4] = f"{analyze_only}./main --mode=batch --saveDirName={saveDirName} --runs={num_exps} --trncyclog=false --tstcyclog=false \
                        {cmd_string} \n"
            lines[-3] = f"python -u Post_analyses.py {saveDirName} cmd\n"
            lines[-2] = f"python -u read_task_parameters_into_csv.py {saveDirName} cmd\n"
            lines[-1] = f"python cross_pair_analysis.py {output_dir} cmd"
            # add mode batch
            f.writelines(lines)
            [print(l) for l in lines]


        # Run sbatch file
        s = subprocess.run([f'sbatch {jobfile}'], shell=True)
        print(s)
        time.sleep(1)
        return

    # loop(bools = bools + [False], values = values + [None], depth = depth + 1) # skip if SetAveL = False

    for param_val in params[depth]: # at each level of parameter
        loop(output_dir, params, param_names,
             values = values + [param_val], depth = depth + 1, **kwargs)

if __name__ == "__main__":
    s = subprocess.run([f'go build'], shell=True)
    s = subprocess.run([f'jupyter nbconvert Post_analyses.ipynb --to python'], shell=True)
    s = subprocess.run([f'jupyter nbconvert read_task_parameters_into_csv.ipynb --to python'], shell=True)
    parser = argparse.ArgumentParser(
            description='Parameter search, Emergent model')
    parser.add_argument('parameter_search_job_name', type=str,
                        action='store', help='Name of the parameter search')
    parser.add_argument('--test',
                        action='store_true', help='whether testing or not')
    parser.add_argument('--analyze_only',
                        action='store_true', help='whether analyze only or not')
    parser.add_argument('--searchvar', type=str,
                        action='store', default='same_diff_flag',
                        help='name of variable to search over')
    parser.add_argument('--boundary_plot',
                        action='store_true', help='whether to do cross_pair or boundary condition plot')
    parser.add_argument('--data_dir', default='./figs',
                        action='store', help='directory to save figures')
    
    args = parser.parse_args()
    print("Arguments", args, flush=True)
    parameter_search_job_name = args.parameter_search_job_name
    if not os.path.exists(args.data_dir):
        os.mkdir(args.data_dir)


    output_dir = f"{args.data_dir}/{parameter_search_job_name}" 
    if not args.analyze_only:
        assert not os.path.exists(f"{output_dir}"), f"The directory {output_dir} already exists \
                                                                    Please use a different job name for the parameter search."

        os.mkdir(output_dir)
        
     

    if args.searchvar == 'same_diff_flag':
        param_names = ["--same_diff_flag"]
        same_diff_flag = ['Same', 'Different']
        params = [same_diff_flag] 
    elif args.searchvar == 'LRateOverAll':
        param_names = ['--LRateOverAll']
        LRateOverAll = [0, .01, .05, .1, .02, .5, 1, 2]
        params = [LRateOverAll]



    loop(output_dir, params = params, param_names = param_names,
                test=args.test, analyze_only = args.analyze_only)

    #while check_if_any_running_sbatch_jobs() == True:
    #    time.sleep(30)
    #if args.boundary_plot:
    #    s = subprocess.run([f'jupyter nbconvert boundary_condition_plots.ipynb --to python'], shell=True)
    #    s = subprocess.run([f'python boundary_condition_plots.py {output_dir} cmd'], shell=True)
    #else:
    #    s = subprocess.run([f'jupyter nbconvert cross_pair_analysis.ipynb --to python'], shell=True)
    #    s = subprocess.run([f'python cross_pair_analysis.py {output_dir} cmd'], shell=True)
