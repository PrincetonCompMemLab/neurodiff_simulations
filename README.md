# README
![simulationexample](https://i.imgur.com/Yjof8ac.png)

Model and analysis code corresponding to Ritvo, Nguyen, Turk-Browne & Norman (2023).

These models have been developed based on the [Emergent](https://github.com/emer/emergent) framework, developed primarily by the CCN lab at UCDavis.

- Steps to Installing and Running the Simulations
    1. [Setting up your `GOPATH` and `GOROOT`](#File-structure)
    2. [Install go](#Installing-Go)
    3. [Install **neurodiff_simulations**](#Install-neurodiff_simulations)
        - Cloning the repository: Run `git clone https://github.com/PrincetonCompMemLab/neurodiff_simulations.git`
        - Build the simulation (do this for `chanales` / `favila` / `schlichting` simulations)
            - e.g. `cd` into `emergent-differentiation/color_diff`
            - run `go build`
        - Run the simulations:
            - run `./main`
    4. [Reproducing the plots](#Reproducing-the-plots)

## Assumptions

- OS: We can guarantee performance on Unix-based platforms such as MacOS and Linux.
  Performance on Windows operating systems is not guaranteed.
- Any version starting 1.13 should work. We used Go 1.18
- To run our analyses, we use the [Slurm](https://slurm.schedmd.com/documentation.html) scheduling system.

### File structure:

You need to set the `GOPATH` and `GOROOT`.

- `GOROOT` is where the go executable is
- `GOPATH` is where to install go packages.

Note that `go` forces the separation of `GOPATH` and `GOROOT`, so they can't be the same directory. This README will assume you are using a folder named `go/` for `GOROOT` and `gocode/` for for `GOPATH`.

1. Add the following lines to `~/.bashrc`

    ```
    export GOROOT="PATH/TO/CODE/go"
    export GOPATH="PATH/TO/CODE/gocode"
    export PATH=$GOROOT/bin:$PATH

    ```

    - replace `"PATH/TO/CODE/"` with the paths to the actual paths of your go installs
2. Make sure to run:

    ```
    source ~/.bashrc

    ```

# Installing Go

## Download Go

The first step to setting up emergent is to download Go. Pick your favored binary release [here](https://golang.org/dl/), download it, and run it. The MSI installers for Windows and Mac do all the work for you.

You should download versions 1.13 or later.

### Downloading Go on the command line

To download go using the command line, use `wget` and then the download link. Make sure to download the tar.gz file. For example, if you are downloading Go 1.14 for Linux OS, you can run

```
wget https://dl.google.com/go/go1.14.1.linux-amd64.tar.gz

```

The above command downloads a .tar.gz file wherever you ran the command from. You probably want to run it from /PATH/TO/CODE, or else move the tar.gz file there after you download it.

## Install Go

The command `wget https://dl.google.com/go/go1.14.1.linux-amd64.tar.gz` downloaded a .tar.gz file wherever you ran it. Move it to where you want the GOPATH directory to be (using `mv`). Then unzip the file:

```
tar -xzf go1.14.1.linux-amd64.tar.gz

```

That should have created a folder called `go/`. Change the name to be whatever you called the `GOPATH` directory (i.e. gocode/).

Then make the directory for the `GOROOT`:

```
mkdir PATH/TO/CODE/go

```

Now you should have two folders that match `GOPATH` and `GOROOT`

## Test Your Go Installation

You might want to make sure you installed Go successfully. To do this:

1. Create a folder `$GOROOT/src/hello`
    1. Then create a file named `hello.go` with the following content:

    ```
    package main

    import "fmt"

    func main() {
    	fmt.Printf("hello, world\\n")
    }

    ```

2. 'Build' the script by executing the command:

    ```
    go build

    ```

    within your `hello` folder. This creates a `hello.exe` file in your directory that you can then run to execute the code.


3. Run `hello.exe`.  

- If the command returns `hello, world`, you're golden. If not, try running `echo $PATH` to see if `GOROOT` and `GOPATH` were added correctly.

## Install neurodiff_simulations

- Clone the github repo:
    - Run `git clone https://github.com/PrincetonCompMemLab/neurodiff_simulations.git`
        - Note: make sure to point to correct public repo
        - Note: make sure that the master / main branches are the right branch
- Build the simulation (do this for `chanales` / `favila` / `schlichting` simulations)
    - `cd` into `neurodiff_simulations`
    - run `go build`
- Run the simulations:
    - run `./main`

# Reproducing the plots

- Figures 4, 8, 10: The `figs.py` script in each simulation directory will generate figures corresponding to the simulations of the Chanales et al., Favila et al., and Schlichting et al. studies
    - In each directory, run `python figs.py output_folder_name --data_dir=/PATH/TO/OUTPUT_FOLDER/`
        - Replace `/PATH/TO/OUTPUT_FOLDER/` and `output_folder_name` with the path to store the output folder and the name of the output folder, respectively
        - This applies for all `figs.py` files
        - `figs.py` will create figures in this directory: `/PATH/TO/OUTPUT_FOLDER/output_folder_name/results`
        - If `data_dir` is not specified, the script will save the figures in a subdirectory of the current directory called `./figs`
- Figure 6B:
    - In each simulation directory, run `python figs.py output_folder_name --data_dir=/PATH/TO/OUTPUT_FOLDER/ --searchvar=LRateOverAll`
        - `figs.py` will create figures in this directory: `/PATH/TO/OUTPUT_FOLDER/output_folder_name/results`
- Figure 6A:
    - `figs.py` will create a plot of the learning rate vs. within-pair correlation before and after learning for each condition (e.g. 0/6, 1/6, 2/6, 3/6, 4/6, and 5/6 are the learning conditions of the Chanales et al. study)
    - You can make `figs.py` make this plot for each learning condition by doing the following:
        - For the Chanales et al. simulation, add the flag `--HiddNumOverlapUnits=i` where i is in [0,1,2,3,4,5] to the line `lines[-4] = f"{analyze_only}./main --mode=batch --saveDirName={saveDirName} --runs={num_exps} --trncyclog=false --tstcyclog=false {cmd_string} \n"` in `figs.py`
        - For the Favila et al. simulation, add the flag `--same_diff_flag=i` where i is in ['Same', 'Different'] to the line `lines[-4] = f"{analyze_only}./main --mode=batch --saveDirName={saveDirName} --runs={num_exps} --trncyclog=false --tstcyclog=false {cmd_string} \n"` in `figs.py`
            - the flag is case sensitive !
        -  For the Schlichting et al. simulation, add the flag `--blocked_interleave_flag=i` where i is in ['Blocked', 'Interleave'] to the line `lines[-4] = f"{analyze_only}./main --mode=batch --saveDirName={saveDirName} --runs={num_exps} --trncyclog=false --tstcyclog=false {cmd_string} \n"` in `figs.py`
    - run `python figs.py output_folder_name --data_dir=/PATH/TO/OUTPUT_FOLDER/ --searchvar=LRateOverAll`
