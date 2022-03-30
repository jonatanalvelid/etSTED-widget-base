-----
**NOTE**

See [etSTED-widget](https://github.com/jonatanalvelid/etSTED-widget) for a full description of the functionality, and detailed explanation of the provided detection pipelines and coordinate transformations.

-----

# etSTED-widget-base

Generic event-triggered imaging widget and controller, a generalization of the event-triggered STED widget provided in ImSwitch and presented in Alvelid et al. 2022. Use for implementation in other Python-based microscope control software, or as a basis for implementation in non-compatible microscope control software solutions. Alternatively, use the provided widget and controller as a standalone widget to control event-triggered imaging, if an overarching intersoftware communication solution is present and implementable with this widget. Beware that an intersoftware communication solution can slow down the time between fast image and analysis, as well as detected event and scan initiation due to intersoftware communication delays.

## Installation
To run the etSTED-widget as a standalone widget, or to implement it in your own microscope control software, install the dependencies by running the following commands in either conda or pip from the source repository and in case of the pip-version in the virtual environment of choice. Python 3.9 or later is tested and recommended. 
```
conda env create -f environment.yml
```

```
pip install -r requirements.txt
```
