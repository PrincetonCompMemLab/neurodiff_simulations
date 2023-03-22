import subprocess
import argparse
import loop
import os
import time

if __name__ == "__main__":
    #s = subprocess.run([f'go build'], shell=True)
    #s = subprocess.run([f'jupyter nbconvert Post_analyses.ipynb --to python'], shell=True)
    parser = argparse.ArgumentParser(
            description='Parameter search, Emergent model')
    parser.add_argument('parameter_search_job_name', type=str,
                        action='store', help='Name of the parameter search')
    parser.add_argument('num_subjs', type=int,
                        action='store', help='Name of the parameter search')
    
    
    args = parser.parse_args()
    print("Arguments", args)
    parameter_search_job_name = args.parameter_search_job_name
    cluster = loop.get_current_cluster()
    if b'della' in cluster:
        data_dir = "/scratch/qanguyen/color_diff"
    elif b'spock' in cluster:
        data_dir = "/scratch/vej/color_diff"
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

    # loop.check_if_job_running(21318575)
    output_dir = f"{data_dir}/{parameter_search_job_name}"

    assert not os.path.exists(f"{output_dir}"), f"The directory batch/{parameter_search_job_name} already exists \
                                                                    #Please use a different job name for the parameter search."

    os.mkdir(output_dir)
    jobfile = "parametersearch.sh"
    loop.get_email_address(jobfile)
    for subj in range(args.num_subjs):
        subject_output_dir = f"{output_dir}/subject{subj}"
        os.makedirs(subject_output_dir)
        loop.loop(subject_output_dir, test=True)

    while loop.check_if_any_running_sbatch_jobs() == True:
        time.sleep(30)
    # Once there are no more jobs, run concat_cross_batch_csv()
    s = subprocess.run([f'jupyter nbconvert cross_pair_analysis.ipynb --to python'], shell=True)
    for subj in range(args.num_subjs):
        s = subprocess.run([f'python cross_pair_analysis.py {output_dir}/subject{subj} cmd'], shell=True)
    # concat_cross_batch_csv(output_dir)
