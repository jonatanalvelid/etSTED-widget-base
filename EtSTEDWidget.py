import os

import napari
import pyqtgraph as pg
from napari.utils.translations import trans
from PyQt5 import QtGui, QtWidgets, QtCore


class EtSTEDWidget(QtWidgets.QWidget):
    """ Widget for controlling the etSTED implementation. """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        print('Initializing etSTED widget')
        
        # generate dropdown list for analysis pipelines
        self.analysisPipelines = list()
        self.analysisPipelinePar = QtGui.QComboBox()
        # generate dropdown list for coordinate transformations
        self.transformPipelines = list()
        self.transformPipelinePar = QtGui.QComboBox()
        # generate dropdown list for fast imaging detectors
        self.fastImgDetectors = list()
        self.fastImgDetectorsPar = QtGui.QComboBox()
        self.fastImgDetectorsPar_label = QtGui.QLabel('Fast detector')
        self.fastImgDetectorsPar_label.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignBottom)
        # generate dropdown list for fast imaging lasers
        self.fastImgLasers = list()
        self.fastImgLasersPar = QtGui.QComboBox()
        self.fastImgLasersPar_label = QtGui.QLabel('Fast laser')
        self.fastImgLasersPar_label.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignBottom)
        # add all experiment modes in a dropdown list
        self.experimentModes = ['Experiment','TestVisualize','TestValidate']
        self.experimentModesPar = QtGui.QComboBox()
        self.experimentModesPar_label = QtGui.QLabel('Experiment mode')
        self.experimentModesPar_label.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignCenter)
        self.experimentModesPar.addItems(self.experimentModes)
        self.experimentModesPar.setCurrentIndex(0)
        # create lists for current pipeline parameters: labels and editable text fields
        self.param_names = list()
        self.param_edits = list()
        # create buttons for initiating the event-triggered imaging and loading the pipeline
        self.initiateButton = QtWidgets.QPushButton('Initiate etSTED')
        self.initiateButton.setSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Expanding)
        self.loadPipelineButton = QtWidgets.QPushButton('Load pipeline')
        # create buttons for calibrating coordinate transform, recording binary mask, loading scan params
        self.coordTransfCalibButton = QtWidgets.QPushButton('Transform calibration')
        self.recordBinaryMaskButton = QtWidgets.QPushButton('Record binary mask')
        self.loadScanParametersButton = QtWidgets.QPushButton('Load scan parameters')
        # creat button for unlocking any softlock happening
        self.setBusyFalseButton = QtWidgets.QPushButton('Unlock softlock')
        # create check box for endless running mode
        self.endlessScanCheck = QtGui.QCheckBox('Endless')
        # create editable fields for binary mask calculation threshold and smoothing
        self.bin_thresh_label = QtGui.QLabel('Bin. threshold')
        self.bin_thresh_label.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignBottom)
        self.bin_thresh_edit = QtGui.QLineEdit(str(10))
        self.bin_smooth_label = QtGui.QLabel('Bin. smooth (px)')
        self.bin_smooth_label.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignBottom)
        self.bin_smooth_edit = QtGui.QLineEdit(str(2))
        # help widget for coordinate transform
        self.coordTransformWidget = CoordTransformWidget(*args, **kwargs)

        # help widget for showing images from the analysis pipelines, i.e. binary masks or analysed images in live
        self.analysisHelpWidget = AnalysisWidget(*args, **kwargs)

        self.grid = QtGui.QGridLayout()
        self.setLayout(self.grid)

        # initialize widget controls
        currentRow = 0

        self.grid.addWidget(self.initiateButton, currentRow, 0)
        self.grid.addWidget(self.endlessScanCheck, currentRow, 1)
        self.grid.addWidget(self.experimentModesPar_label, currentRow, 2)
        self.grid.addWidget(self.experimentModesPar, currentRow, 3)

        currentRow += 1

        self.grid.addWidget(self.bin_smooth_label, currentRow, 0)
        self.grid.addWidget(self.bin_smooth_edit, currentRow, 1)
        self.grid.addWidget(self.bin_thresh_label, currentRow, 2)
        self.grid.addWidget(self.bin_thresh_edit, currentRow, 3)

        currentRow += 1

        self.grid.addWidget(self.loadPipelineButton, currentRow, 0)
        self.grid.addWidget(self.analysisPipelinePar, currentRow, 1)
        self.grid.addWidget(self.transformPipelinePar, currentRow, 2)
        self.grid.addWidget(self.coordTransfCalibButton, currentRow, 3)

        currentRow += 1

        self.grid.addWidget(self.loadScanParametersButton, currentRow, 2)
        self.grid.addWidget(self.recordBinaryMaskButton, currentRow, 3)

        currentRow +=1

        self.grid.addWidget(self.fastImgDetectorsPar_label, currentRow, 2)
        self.grid.addWidget(self.fastImgDetectorsPar, currentRow, 3)

        currentRow += 1

        self.grid.addWidget(self.fastImgLasersPar_label, currentRow, 2)
        self.grid.addWidget(self.fastImgLasersPar, currentRow, 3)

        currentRow +=1 

        self.grid.addWidget(self.setBusyFalseButton, currentRow, 3)

    def initParamFields(self, parameters: dict, params_exclude: list):
        """ Initialized event-triggered analysis pipeline parameter fields. """
        # remove previous parameter fields for the previously loaded pipeline
        for param in self.param_names:
            self.grid.removeWidget(param)
        for param in self.param_edits:
            self.grid.removeWidget(param)

        # initiate parameter fields for all the parameters in the pipeline chosen
        currentRow = 3
        
        self.param_names = list()
        self.param_edits = list()
        for pipeline_param_name, pipeline_param_val in parameters.items():
            if pipeline_param_name not in params_exclude:
                # create param for input
                param_name = QtGui.QLabel('{}'.format(pipeline_param_name))
                param_value = pipeline_param_val.default if pipeline_param_val.default is not pipeline_param_val.empty else 0
                param_edit = QtGui.QLineEdit(str(param_value))
                # add param name and param to grid
                self.grid.addWidget(param_name, currentRow, 0)
                self.grid.addWidget(param_edit, currentRow, 1)
                # add param name and param to object list of temp widgets
                self.param_names.append(param_name)
                self.param_edits.append(param_edit)

                currentRow += 1

    def setAnalysisPipelines(self, analysisDir):
        """ Set combobox with available analysis pipelines to use. """
        for pipeline in os.listdir(analysisDir):
            if os.path.isfile(os.path.join(analysisDir, pipeline)):
                pipeline = pipeline.split('.')[0]
                self.analysisPipelines.append(pipeline)
        self.analysisPipelinePar.addItems(self.analysisPipelines)
        self.analysisPipelinePar.setCurrentIndex(0)

    def setTransformations(self, transformDir):
        """ Set combobox with available coordinate transformations to use. """
        for transform in os.listdir(transformDir):
            if os.path.isfile(os.path.join(transformDir, transform)):
                transform = transform.split('.')[0]
                self.transformPipelines.append(transform)
        self.transformPipelinePar.addItems(self.transformPipelines)
        self.transformPipelinePar.setCurrentIndex(0)

    def setFastDetectorList(self, detectorNames):
        """ Set combobox with available detectors to use for the fast method. """
        self.fastImgDetectors = detectorNames
        self.fastImgDetectorsPar.addItems(self.fastImgDetectors)
        self.fastImgDetectorsPar.setCurrentIndex(0)

    def setFastLaserList(self, laserNames):
        """ Set combobox with available lasers to use for the fast method. """
        self.fastImgLasers = laserNames
        self.fastImgLasersPar.addItems(self.fastImgLasers)
        self.fastImgLasersPar.setCurrentIndex(0)

    def launchHelpWidget(self, widget, init=True):
        """ Launch the help widget. """
        if init:
            widget.show()
        else:
            widget.hide()


class AnalysisWidget(QtWidgets.QWidget):
    """ Pop-up widget for the live analysis images or binary masks. """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.imgVbWidget = pg.GraphicsLayoutWidget()
        self.imgVb = self.imgVbWidget.addViewBox(row=1, col=1)

        self.img = pg.ImageItem(axisOrder = 'row-major')
        self.img.translate(-0.5, -0.5)

        self.imgVb.addItem(self.img)
        self.imgVb.setAspectLocked(True)

        self.info_label = QtGui.QLabel('<image info>')
        
        # generate GUI layout
        self.grid = QtGui.QGridLayout()
        self.setLayout(self.grid)

        self.grid.addWidget(self.info_label, 0, 0)
        self.grid.addWidget(self.imgVbWidget, 1, 0)


class CoordTransformWidget(QtWidgets.QWidget):
    """ Pop-up widget for the coordinate transform between the two etSTED modalities. """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loadLoResButton = QtWidgets.QPushButton('Load low-res calibration image')
        self.loadHiResButton = QtWidgets.QPushButton('Load high-res calibration image')
        self.saveCalibButton = QtWidgets.QPushButton('Save calibration')
        self.resetCoordsButton = QtWidgets.QPushButton('Reset coordinates')

        self.napariViewerLo = EmbeddedNapari()
        self.napariViewerHi = EmbeddedNapari()

        # add points layers to the viewer
        self.pointsLayerLo = self.napariViewerLo.add_points(name="lo_points", symbol='ring', size=20, face_color='green', edge_color='green')
        self.pointsLayerTransf = self.napariViewerHi.add_points(name="transf_points", symbol='cross', size=20, face_color='red', edge_color='red')
        self.pointsLayerHi = self.napariViewerHi.add_points(name="hi_points", symbol='ring', size=20, face_color='green', edge_color='green')
        

        # generate GUI layout
        self.grid = QtGui.QGridLayout()
        self.setLayout(self.grid)
    
        currentRow = 0
        self.grid.addWidget(self.loadLoResButton, currentRow, 0)
        self.grid.addWidget(self.loadHiResButton, currentRow, 1)
        
        currentRow += 1
        self.grid.addWidget(self.napariViewerLo.get_widget(), currentRow, 0)
        self.grid.addWidget(self.napariViewerHi.get_widget(), currentRow, 1)

        currentRow += 1
        self.grid.addWidget(self.saveCalibButton, currentRow, 0)
        self.grid.addWidget(self.resetCoordsButton, currentRow, 1)


class EmbeddedNapari(napari.Viewer):
    """ Napari viewer to be embedded in non-napari windows. Also includes a
    feature to protect certain layers from being removed when added using
    the add_image method. Copied from ImSwitch (github.com/kasasxav/ImSwitch).
    """

    def __init__(self, *args, show=False, **kwargs):
        super().__init__(*args, show=show, **kwargs)

        # Monkeypatch layer removal methods
        oldDelitemIndices = self.layers._delitem_indices

        def newDelitemIndices(key):
            indices = oldDelitemIndices(key)
            for index in indices[:]:
                layer = index[0][index[1]]
                if hasattr(layer, 'protected') and layer.protected:
                    indices.remove(index)
            return indices

        self.layers._delitem_indices = newDelitemIndices

        # Make menu bar not native
        self.window._qt_window.menuBar().setNativeMenuBar(False)

        # Remove unwanted menu bar items
        menuChildren = self.window._qt_window.findChildren(QtWidgets.QAction)
        for menuChild in menuChildren:
            try:
                if menuChild.text() in [trans._('Close Window'), trans._('Exit')]:
                    self.window.file_menu.removeAction(menuChild)
            except Exception:
                pass

    def add_image(self, *args, protected=False, **kwargs):
        result = super().add_image(*args, **kwargs)

        if isinstance(result, list):
            for layer in result:
                layer.protected = protected
        else:
            result.protected = protected
        return result

    def get_widget(self):
        return self.window._qt_window

    def addItem(self, item):
        item.attach(self,
                    canvas=self.window.qt_viewer.canvas,
                    view=self.window.qt_viewer.view,
                    parent=self.window.qt_viewer.view.scene,
                    order=1e6 + 8000)
