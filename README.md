-----
**NOTE**

See [etSTED-widget](https://github.com/jonatanalvelid/etSTED-widget) for a full description of the functionality, and detailed explanation of the provided detection pipelines and coordinate transformations.

See [ImSwitch](https://github.com/kasasxav/ImSwitch) for a full control software with this etSTED widget and controller integrated, as used in Alvelid et al. 2022. 

-----

# etSTED-widget-base

Generic event-triggered imaging widget and controller, a generalization of the event-triggered STED widget provided in ImSwitch and presented in Alvelid et al. 2022. Use for implementation in other Python-based microscope control software, or as a basis for implementation in non-compatible microscope control software solutions. Alternatively, use the provided widget and controller as a standalone widget to control event-triggered imaging, if an overarching intersoftware communication solution is present and implementable with this widget. Beware that an intersoftware communication solution can slow down the time between fast image and analysis, as well as detected event and scan initiation due to intersoftware communication delays.

## Installation
To run the etSTED-widget as a standalone widget, or to implement it in your own microscope control software, install the dependencies by running the following command in pip from the source repository and in the virtual environment of choice. Python 3.7 or later is tested and recommended. 

```
pip install -r requirements.txt
```

## Development and implementation in control software
In order to implement this widget into your control software of choice, change the lines that are TODO-marked and commented to the control implementations specific to your control software. This includes for example signals for receiving the recently acquired fast imaging images, for knowing when the scanned image has finished, or for turning on/off the fast imaging laser. 
