# Leabra Color Diff Example

(description)

# Running the model

First, see [Wiki Install](https://github.com/emer/emergent/wiki/Install) for installation instructions, which includes how to build the model code in this directory, which will make an executable named `ra25` that you run from a terminal command line:

```bash
./ra25
```

You can also run the [Python](https://github.com/emer/leabra/blob/master/python/README.md) version of the model by following those instructions.

## Basic running and graphing

The basic interface has a toolbar across the top, a set of `Sim` fields on the left, and a tab bar with the network view and various log graphs on the right.  The toolbar actions all have tooltips so just hover over those to see what they do.  To speed up processing, you'll want to either turn off the `ViewOn` flag or switch to a graph tab instead of the `NetVew` tab.

You can change the speed at which the model updates by changing the TrainUpdt or TestUpdt parameter on the left of the Gui. For instance, the model can update at the cycle level (slowest) or the epoch level (fastest).

## Testing

The Test* buttons allow you to test items, including a specifically-chosen item, without having any effect on the ongoing training.  This is one advantage of the [Env](https://github.com/emer/emergent/wiki/Env) interface, which keeps all the counters associated with training and testing separate.

The NetView will show cycle-by-cycle updates during testing, and you can see the temporal evolution of the activities in the `TstCycPlot`.  If you do `TestAll` and look at the `TstTrlPlot` you can see the current performance on every item.  Meanwhile, if you click on the `TstTrlLog` button at the left, you can see the input / output activations for each item in a TableView, and the `TstErrLog` button likewise shows the same thing but filtered to only show those trials that have an error.  `TstErrStats` computes some stats on those error trials -- not super meaningful here but could be in other more structured environments, and the code that does all this shows how to do all of this kind of data analysis using the [etable.Table](https://github.com/emer/etable) system, which is similar to the widely-used pandas DataFrame structure in Python, and is the updated version of the `DataTable` from C++ emergent.

## Parameters

Clicking on the `Params` button will pull up a set of parameters, the design and use of which are explained in detail on the wiki page: [Params](https://github.com/emer/emergent/wiki/Params).  When you hit `Init`, the `Base` ParamSet is always applied, and then if you enter the name of another ParamSet in the `ParamSet` field, that will then be applied after the Base, thereby overwriting those base default params with other ones to explore.

To see any non-default parameter settings, the `Non Def Params` button in the NetView toolbar will show you those, and the `All Params` button will show you *all* of the parameters for every layer and projection in the network.  This is a good way to see all the parameters that are available.

To determine the real performance impact of any of the params, you typically need to collect stats over multiple runs.  To see how this works, try the following:

* Click on the `RunPlot` tab and hit `ResetRunLog` for good measure.
* Init with `ParamSet` = empty, and do `Train` and let it run all 10 runs.  By default, it plots the epoch at which the network first hit 0 errors (`FirstZero`), which is as good a measure as any of overall learning speed.
* When it finishes, you can click on the `RunStats` Table to see the summary stats for FirstZero and the average over the last 10 epochs of `PctCor`, and it labels this as using the Base params.
* Now enter `NoMomentum` in the `ParamSet`, `Init` and `Train` again.  Then click on the `RunStats` table button again (it generates a new one after every complete set of runs, so you can just close the old one -- it won't Update to show new results).  You can now directly compare e.g., the Mean FirstZero Epoch, and see that the `Base` params are slightly faster than `NoMomentum`.
* Now you can go back to the params, duplicate one of the sets, and start entering your own custom set of params to explore, and see if you can beat the Base settings!  Just click on the `*params.Sel` button after `Network` to get the actual parameters being set, which are contained in that named `Sheet`.
* Click on the `Net` button on the left and then on one of the layers, and so-on into the parameters at the layer level (`Act`, `Inhib`, `Learn`), and if you click on one of the `Prjn`s, you can see parameters at the projection level in `Learn`.  You should be able to see the path for specifying any of these params in the Params sets.
* We are planning to add a function that will show you the path to any parameter via a context-menu action on its label..

## Parameter Searching

#### one time:

* Add the following line to your `~/.bashrc`
  * If you're on della add `export CLUSTER_NAME='della'`
  * If you're on spock add `export CLUSTER_NAME='spock'`
  * Do `source ~/.bashrc`
  * You only need to do this once

#### for each parameter to test over:
1. Create a flag variable in the `CmdArgs()` function in `color_diff.go`
```go
Variable := CreateFlagVariable("Flag_Name", FLAG_VARIABLE_TYPE, DEFAULT_VALUE_OF_FLAG, "variable description").(*FLAG_VARIABLE_TYPE)
```
  * NOTES:
     * The value of `DEFAULT_VALUE_OF_FLAG` should have type `FLAG_VARIABLE_TYPE`
     * The variable `Variable` will be a pointer to the value determined by the default (if not given on the command line) or the value on the command line

     * for instance, create a flag name HiddNumOverlap Units, like this:
  ```
  CreateFlagVariable("HiddNumOverlapUnits", "int", 2, "numer of overlapping units")
```
     * You sometimes don't need to set the default value of the parameter, if the default value is already in parameter search. You might also not need to name the variable
     * In this case, the `DEFAULT_VALUE_OF_FLAG` should be `nil`

2. In the function `ConfigParamValues()`, edit the Param Sheets with two lines.
```go
Layer_Params := ss.Params.SetByName("TaskColorRecall").SheetByName("Network").SelByName("Layer").Params
ss.SetParamSheetValuefromParamValues(Layer_Params, "Layer.Learn.AvgL.AveLFix", "Layer_ColorRecall_Learn_AvgL_AveLFix")
```  
  * The first argument is the param sheet to be edited (a dictionary). You should use the same flag variable name you used when you created the flag variable in #1 (i.e. HiddNumOverlapUnits, above)

  * In the second line,
    * the function `ss.SetParamSheetValuefromParamValues` edits the correct sheet
    * The second argument is the key in the dictionary to be edited
    * The third argument is the name of the flag that holds the value that we want to input into the param sheet
    * Note that if we have not set a flag, then `ss.SetParamSheetValuefromParamValues` does not change the param sheets

3.  In the `loop.py` script, add the parameters to search over.
  1. add the parameter names to be searched over in `param_names`
  ```python
  param_names = ["--Hidden_ColorRecall_Layer_OscAmnt", "--Face_ColorRecall_Layer_OscAmnt", "--Output_ColorRecall_Layer_OscAmnt"]
  ```
  2.  For each parameter you listed in 3.1, add the ranges to be searched over into `params` :
```python
param_names = ["--Hidden_ColorRecall_Layer_OscAmnt", "--Face_ColorRecall_Layer_OscAmnt", "--Output_ColorRecall_Layer_OscAmnt"]
Hidden_ColorRecall_Layer_OscAmnt = [0.1,0.15,0.2,0.25, 0.3,0.35,0.4]
Face_ColorRecall_Layer_OscAmnt = [0.1,0.2,0.3,0.4, 0.5,0.6,0.7]
Output_ColorRecall_Layer_OscAmnt = [0.1,0.15,0.2,0.25, 0.3,0.35,0.4]
```
  3. Add those values taht you are searching over to the params array:
```
 params = [Hidden_ColorRecall_Layer_OscAmnt,Face_ColorRecall_Layer_OscAmnt, Output_ColorRecall_Layer_OscAmnt]
```

To run the search, you need to name the search you are doing. So you would run something like

```
nohup python loop.py NAME_OF_PARAM_SEARCH
```

That creates a folder, /scratch/vej/color_diff/NAME_OF_PARAM_SEARCH/

that holds all the individual directories.

You can then sort through them with cross_batch_comparison.ipynb

## Running from command line

Type this at the command line:
```bash
./colordiff -help
```

To see a list of args that you can pass -- passing any arg will cause the model to run without the gui, and save log files and, optionally, final weights files for each run.

For example,

```
./color_diff --tsttrllog=true --runs=20
```
Would run 20 runs, and would run a test trial log run as well.

```
./color_diff --tsttrllog=true --trncyclog=true --runs=20
```
Would additionally include a log of the cycle activation across training trials.

### Running the analysis from the command line (on the cluster)
* Install ray: Run `pip install ray` in the environment you're using to run `Post_analyses.py`
  * You only need to do this once
* If running in spock, this script assumes that all the data files are at `/scratch/vej/color_diff/`
  * If running in della, this script assumes that all the data files are at `/scratch/vej/color_diff/`
* If applicable, activate the correct environment (the same environment used by Post_analyses.ipynb)
* Run `sbatch analyze_batch.sh name_of_the_batch`

# Code organization and notes

Most of the code is commented and should be read directly for how to do things.  Here are just a few general organizational notes about code structure overall.

* Good idea to keep all the code in one file so it is easy to share with others, although fine to split up too if it gets too big -- e.g., logging takes a lot of space and could be put in a separate file.

* In Go, you can organize things however you want -- there are no constraints on order in Go code.  In Python, all the methods must be inside the main Sim class definition but otherwise order should not matter.

* The GUI config and elements are all optional and the -nogui startup arg, along with other args, allows the model to be run without the gui.

* If there is a more complex environment associated with the model, always put it in a separate file, so it can more easily be re-used across other models.

* The params editor can easily save to a file, default named "params.go" with name `SavedParamsSets` -- you can switch your project to using that as its default set of params to then easily always be using whatever params were saved last.
* TaskColorStudy is plus only. This is done by 1) at the beginning of the task, add ss.Net.LearningMP = 0; for plus only, or ss.Net.LearningMP = 1; for minus plus. 2) If you want a higher amount of hebbian leanring, within the parameter sheets, change SetLLrn to be True, and set LLrn to be the amount of hebbian learning, and set MLrn to be 1-Llrn (?)
* To add OscAmnt to Gui
  * uncomment OscAmnt in Sim in line 244
  * uncomment the code that overrides OscAmnt in ss.Params with GUI's OscAmnt value in each task (i.e. TaskColorWOOsc and TaskColorStudy)

# Running Jupyter Notebooks on the cluster
* On the cluster,  run `sbatch jupyter_start.sh` and wait a while for the job to be assigned
* A file called `jupyter-notebook-"job_number".sh` will be created.

* If you run `cat jupyter-notebook-"job_number".sh` something like this will show up

```
For more info and how to connect from windows,
see https://docs.ycrc.yale.edu/clusters-at-yale/guides/jupyter/



MacOS or linux terminal command to create your ssh tunnel
ssh -N -L 9664:redshirt-n17:9664 qanguyen@spock.princeton.edu

Windows MobaXterm info
Forwarded port:same as remote port
Remote server: redshirt-n17
Remote port: 9664
SSH server: spock.princeton.edu
SSH login: qanguyen
SSH port: 22

Use a Browser on your local machine to go to:
localhost:9664  (prefix w/ https:// if using password)

[I 22:46:14.199 NotebookApp] Serving notebooks from local directory: /mnt/bucket/people/qanguyen/gocode/src/github.com/emer/private-leabra/examples/color_diff
[I 22:46:14.200 NotebookApp] 0 active kernels
[I 22:46:14.200 NotebookApp] The Jupyter Notebook is running at: http://redshirt-n17:9664/?token=8b1eec980d5c39f20602d65c983d6eabac8b291e10a55dae
[I 22:46:14.200 NotebookApp] Use Control-C to stop this server and shut down all kernels (twice to skip confirmation).
[C 22:46:14.207 NotebookApp]



    Copy/paste this URL into your browser when you connect for the first time,
    to login with a token:
        http://redshirt-n17:9664/?token=8b1eec980d5c39f20602d65c983d6eabac8b291e10a55dae
```

* Copy the line `ssh -N -L 9664:redshirt-n17:9664 qanguyen@spock.princeton.edu` to your local terminal. You might not see the number 9664, but just copy that whole line into your terminal.

* Then in a browser go to the URL `localhost:9664` (again you won't see the same number but it'll tell you exactly what to copy)

* Then it'll ask you for a password token, so put in a token like `8b1eec980d5c39f20602d65c983d6eabac8b291e10a55dae` or whatever it tells you

* Then you should be able to use jupyter in our browser, but the notebook is on the cluster!

* `jupyter_start.sh` creates a jupyter notebook server called 9664 on the cluster.
  * `ssh` creates a connection between your local server 9664 and the cluster server 9664, allowing you to use it


  # Best versions
  * '2021_11_23_favila' :
     * now with hidden_output learning.
  
