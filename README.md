# DecisionTree
Decision tree classifier using a real-word data-set called Census-Income Data Set.

## Setting up the virtual environment
Make sure you are inside this project's directory. For this project assignment we will be using vanv because 
venv creates virtual environments in the shell that are fresh and sandboxed, with user-installable libraries, and it's multi-python safe.

venv is a package shipped with python 3. If python 3 is install properly, you already have this virtualization functionality: 

Making sure you are in the project directory, install a new virtual environment for use. Further indicate the folder were all dependencies will exist.:
*We name the file containing the dependencies 'venv'*
```bash
python3 -m venv ./venv
```

Now that we have a new virtual environment installed, it needs to be activated. 
```bash
source ./venv/bin/activate
```
*You should now see '(venv)' displayed to the left of your primary prompt string.*

Verify that you are using python3 and pip from 'venv' file *(our virtual machine source)* 
```bash
which python3
```
```bash
which pip
```

Finished developing for this specific project. To exit virtual machine:
```bash
deactivate
```

For any reason if you need to get rid of a virtual machine entirely, just delete the folder created during installation and activation of the virtual machine *('venv')*.

### Install packages your project depends on 
Virtual machine must be active for the following commands to work properly withg virtual machines.

Install packages the command below.
```bash
pip install [PACKAGE]
```

Uninstall packages the command below.
```bash
pip uninstall [PACKAGE]
```

Check packages already installed.
```bash
pip list
```

Port the dependencies to a file called 'requirement.txt' for others to use in their own virtual environment
```bash
pip freeze > requirements.txt
```

### Setting up your own venv with someone else's packages. 
Virtual environment source directories should be excluded from repositories and the 'requirements.txt' should be used to set up a newly created virtual environment. This is because of errors that can/probably will arise when using someone else's environment. One being the hardcoded paths issues to arise next time commands like 'pip' are run.

First follow the 'Setting up the virtual environment' walkthrough above and get virtual environment 'venv' directory correctly set up. Then with the 'requirements.txt' file you cloned/pulled/retrieved run this command to install the same dependencies. 
```bash
pip install -r requirements.txt
```
*This assures that you have the same packages and can run the code accompanying the 'requirements.txt' file*

##