import numpy as np
import argparse
import subprocess
import datetime
import time
import os
import pandas as pd
import glob as glob




def get_email_address(jobfile):
    process = subprocess.Popen(['whoami'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    user, err = process.communicate()

    with open(jobfile, "r") as f:
        lines = f.readlines()
    with open(jobfile, "w") as f:
        if user == b'qanguyen\n':
            lines[6] = "#SBATCH --mail-user=alexn@uni.minerva.edu\n"
        else:
            lines[6] = "#SBATCH --mail-user=vej@princeton.edu\n"
        f.writelines(lines)

def get_current_cluster():
    return subprocess.check_output('echo $CLUSTER_NAME', shell=True)

def get_current_user():
    process = subprocess.Popen(['whoami'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    user, err = process.communicate()

    if user == b'qanguyen\n':
        user = 'qanguyen'
    else:
        user = 'vej'
    return user 

def check_if_any_running_sbatch_jobs():
    user = get_current_user()

    if 'qanguyen' in user:
        user = 'qanguyen'
    else:
        user = 'vej'
        
    process = subprocess.Popen(['squeue', '-u', user], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    running_jobs, err = process.communicate()
    running_jobs = running_jobs.decode("utf-8").split("\n")

    for job in running_jobs:
        # If the job name has color_di in it, then there are still jobs returning
        if 'favila' in job:
            return True

    # Else, there are no more color_diff jobs left
    return False


def check_if_job_running(jobid):

    process = subprocess.Popen(['squeue', '-j', str(jobid)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    running_jobs, err = process.communicate()
    running_jobs = running_jobs.decode("utf-8").split("\n")
    if running_jobs == [""]:
        # Else, there are no more color_diff jobs left
        return False
    else:
        return True


def concat_cross_batch_csv(output_dir):
    for file_type in ['checkpoints', 'results', 'analyses']:
        data_folders = glob.glob(output_dir + "/*")
        if file_type == 'analyses':
            csv_path = [folder + '/fig/' + file_type + '.csv' for folder in data_folders if os.path.isdir(folder)]
        else:
            csv_path = [folder + '/fig/' + file_type + '/' + file_type + '.csv' for folder in data_folders if os.path.isdir(folder)]

        all_df = []
        for d_file in csv_path:
            csv_read = pd.read_csv(d_file)
            all_df.append(csv_read)

        data_dir = f'{output_dir}/all_' + file_type + '.csv'
        all_df = pd.concat(all_df)
        all_df.to_csv(data_dir)
        del(all_df)



def string(boolean):
    if boolean == True:
        return "true"
    else:
        return "false"

def loop(output_dir, params, param_names,
         values = [], depth = 0, **kwargs):

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

        print(f"./favila --mode=batch --saveDirName={saveDirName} \
                     {cmd_string}")
        # Edit sbatch file
        with open(jobfile, "r+") as f:
            lines = f.readlines()

        print(lines)
        with open(jobfile, "w") as f:
            lines[-3] = f"{analyze_only}./favila --mode=batch --saveDirName={saveDirName} --runs={num_exps} --trncyclog=false --tstcyclog=false \
                        {cmd_string} \n"
            lines[-2] = f"python -u Post_analyses.py {saveDirName} cmd\n"
            lines[-1] = f"python -u read_task_parameters_into_csv.py {saveDirName} cmd"

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
    
    args = parser.parse_args()
    print("Arguments", args, flush=True)
    parameter_search_job_name = args.parameter_search_job_name
    cluster = get_current_cluster()
    user = get_current_user()
    if b'della' in cluster:
        data_dir = f"/scratch/{user}/favila"
    elif b'spock' in cluster:
        data_dir = f"/scratch/{user}/favila"
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)


    output_dir = f"{data_dir}/{parameter_search_job_name}"
    if not args.analyze_only:
        assert not os.path.exists(f"{output_dir}"), f"The directory {output_dir} already exists \
                                                                    Please use a different job name for the parameter search."

        os.mkdir(output_dir)
        
    #if you want to change the type of emails you get, change it here:
    jobfile = "parametersearch.sh"
    with open(jobfile, "r") as f:
        lines = f.readlines()
        lines[5] = "#SBATCH --mail-type=ALL\n"

    get_email_address(jobfile)

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

    while check_if_any_running_sbatch_jobs() == True:
        time.sleep(30)
    # Once there are no more jobs, run concat_cross_batch_csv()
    if args.boundary_plot:
        s = subprocess.run([f'jupyter nbconvert boundary_condition_plots.ipynb --to python'], shell=True)
        s = subprocess.run([f'python boundary_condition_plots.py {output_dir} cmd'], shell=True)
    else:
        s = subprocess.run([f'jupyter nbconvert cross_pair_analysis.ipynb --to python'], shell=True)
        s = subprocess.run([f'python cross_pair_analysis.py {output_dir} cmd'], shell=True)
    # concat_cross_batch_csv(output_dir)
