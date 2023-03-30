# Installing emergent
![simulationexample](https://i.imgur.com/Yjof8ac.png)
- List of steps
    - Setting up your `GOPATH` and `GOROOT`
    - Install go
    - Install **emergent-differentiation**
        - Cloning the repository: Run `git clone [https://github.com/PrincetonCompMemLab/emergent-differentiation.git](https://github.com/PrincetonCompMemLab/emergent-differentiation.git)`
        - Build the simulation (do this for `color_diff` / `favila` / `schlichting` simulations)
            - e.g. `cd` into `emergent-differentiation/color_diff`
            - run `go build`
        - Run the simulations:
            - run `./main`

## Assumptions

- OS: Linux / Mac
- Go 1.18

### File structure:

You need to set the `GOPATH` and `GOROOT`.

- `GOROOT` is where the go executable is
- `GOPATH` is where to install go packages.

You'll use `GOPATH` for the actual project folders for your model. Each project will have a directory, so it'll be something like `GOPATH/src/github.com/emer/leabra/examples/PROJECTNAME.`

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


## Download Go

The first step to setting up emergent is to download Go. Pick your favored binary release [here](https://golang.org/dl/), download it, and run it. The MSI installers for Windows and Mac do all the work for you.

You should download versions 1.13 or later.

### Downloading Go on the command line

To download go on the cluster, use `wget`  and then the download link, such as

```
wget https://dl.google.com/go/go1.14.1.linux-amd64.tar.gz

```

In general, to download a file, just run `wget` . The above command downloads a .tar.gz file wherever you are. You probably want to run it from /PATH/TO/CODE, or else move the tar.gz file there after you download it.

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

## Install [**our project**]

- Clone the github repo:
    - Run `git clone [https://github.com/PrincetonCompMemLab/emergent-differentiation.git](https://github.com/PrincetonCompMemLab/emergent-differentiation.git)`
        - Note: make sure to point to correct public repo
        - Note: make sure that the master / main branches are the right branch
- Build the simulation (do this for `color_diff` / `favila` / `schlichting` simulations)
    - `cd` into `emergent-differentiation/color_diff`
    - run `go build`
- Run the simulations:
    - run `./main`

# Reproducing the plots

- Figure 4:
    - python [loop.py](http://loop.py) output_folder_name
- Figure 6:
    -
