import os
import glob
import sys
import importlib
import enum
from collections import deque
from datetime import datetime
from inspect import signature
from tkinter import Tk, filedialog

import h5py
import scipy.ndimage as ndi
import numpy as np
from scipy.optimize import least_squares

# TODO: change this to preferred path for saved logs
_logsDir = os.path.join('C:\\etSTED', 'recordings', 'logs_etsted')
# TODO: change this to path for pipelines and transformation files
_etstedDir = os.path.join('C:\\etSTED', 'imcontrol_etsted')


class EtSTEDController():
    """ Linked to EtSTEDWidget."""

    def __init__(self, widget,  *args, **kwargs):
        self._widget = widget
        
        print('Initializing etSTED controller')

        # folders for analysis pipelines and transformations
        self.analysisDir = os.path.join(_etstedDir, 'analysis_pipelines')
        if not os.path.exists(self.analysisDir):
            os.makedirs(self.analysisDir)
        sys.path.append(self.analysisDir)
        self.transformDir = os.path.join(_etstedDir, 'transform_pipelines')
        if not os.path.exists(self.transformDir):
            os.makedirs(self.transformDir)
        sys.path.append(self.transformDir)
        # set lists of analysis pipelines and transformations in the widget
        self._widget.setAnalysisPipelines(self.analysisDir)
        self._widget.setTransformations(self.transformDir)

        # list of available detectors for fast imaging, get this list from elsewhere in the software
        # TODO: get list of available detectors
        self.detectorList = ['MockCamera']  # mock
        self._widget.setFastDetectorList(self.detectorList)

        # list of available lasers for fast imaging, get this list from elsewhere in the software
        # TODO: get list of available lasers
        self.laserList = ['FastLaser','ScanExcLaser','ScanSTEDLaser']  # mock
        self._widget.setFastLaserList(self.laserList)

        # create a helper controller for the coordinate transform pop-out widget
        self.__coordTransformHelper = EtSTEDCoordTransformHelper(self, self._widget.coordTransformWidget, _logsDir)

        # Connect EtSTEDWidget and communication channel signals
        self._widget.initiateButton.clicked.connect(self.initiate)
        self._widget.loadPipelineButton.clicked.connect(self.loadPipeline)
        self._widget.recordBinaryMaskButton.clicked.connect(self.initiateBinaryMask)
        self._widget.loadScanParametersButton.clicked.connect(self.getScanParameters)
        self._widget.setBusyFalseButton.clicked.connect(self.setBusyFalse)

        # initiate log for each detected event
        self.resetDetLog()
        # initiate pipeline parameter values
        self.resetPipelineParamVals()
        # initiate run parameters
        self.resetRunParams()
        # initiate other parameters and flags used during experiments
        self.initiateFlagsParams()

    def initiateFlagsParams(self):
        # initiate flags and params
        self.__running = False  # run flag
        self.__runMode = RunMode.Experiment  # run mode currently used
        self.__validating = False  # validation flag
        self.__busy = False  # running pipeline busy flag
        self.__bkg = None  # bkg image
        self.__prevFrames = deque(maxlen=10)  # deque for previous fast frames
        self.__prevAnaFrames = deque(maxlen=10)  # deque for previous preprocessed analysis frames
        self.__binary_mask = None  # binary mask of regions of interest, used by certain pipelines, leave None to consider the whole image
        self.__binary_frames = 10  # number of frames to use for calculating binary mask 
        self.__init_frames = 5  # number of frames after initiating etSTED before a trigger can occur, to allow laser power settling etc
        self.__validation_frames = 5  # number of fast frames to record after detecting an event in validation mode
        self.__params_exclude = ['img', 'bkg', 'binary_mask', 'exinfo', 'testmode']  # excluded pipeline parameters when loading param fields

    def initiate(self):
        """ Initiate or stop an etSTED experiment. """
        if not self.__running:
            # detector and laser for fast imaging
            detectorFastIdx = self._widget.fastImgDetectorsPar.currentIndex()
            self.detectorFast = self._widget.fastImgDetectors[detectorFastIdx]
            laserFastIdx = self._widget.fastImgLasersPar.currentIndex()
            self.laserFast = self._widget.fastImgLasers[laserFastIdx]

            # Read GUI params for analysis pipeline
            self.__pipeline_param_vals = self.readPipelineParams()
            # reset general run parameters
            self.resetRunParams()
            # Reset parameter for extra information that pipelines can input and output
            self.__exinfo = None

            # launch help widget, if visualization mode or validation mode
            # Check if visualization mode, in case launch help widget
            experimentModeIdx = self._widget.experimentModesPar.currentIndex()
            self.experimentMode = self._widget.experimentModes[experimentModeIdx]
            if self.experimentMode == 'TestVisualize':
                self.__runMode = RunMode.TestVisualize
            elif self.experimentMode == 'TestValidate':
                self.__runMode = RunMode.TestValidate
            else:
                self.__runMode = RunMode.Experiment
            # check if visualization or validation mode
            if self.__runMode == RunMode.TestValidate or self.__runMode == RunMode.TestVisualize:
                self.launchHelpWidget()
            # load selected coordinate transform
            self.loadTransform()
            self.__transformCoeffs = self.__coordTransformHelper.getTransformCoeffs()
            # connect signals and turn on fast laser
            # TODO: connect signal from update of fast image to running pipeline #xxx.sigUpdateImage.connect(self.runPipeline)
            # TODO: connect signal from end of scan to scanEnded() #xxx.sigScanEnded.connect(self.scanEnded)
            # TODO: turn on laserFast #xxx.lasersManager.laserFast.setEnabled(True)
            self._widget.initiateButton.setText('Stop')
            self.__running = True
        else:
            # disconnect signals and turn off fast laser
            # TODO: disconnect signal from update of fast image to running pipeline #xxx.sigUpdateImage.disconnect(self.runPipeline)
            # TODO: disconnect signal from end of scan to scanEnded() #xxx.sigScanEnded.disconnect(self.scanEnded)
            # TODO: turn off laserFast #xxx.lasersManager.laserFast.setEnabled(False)
            self._widget.initiateButton.setText('Initiate')
            self.resetPipelineParamVals()
            self.resetRunParams()

    def scanEnded(self):
        """ End an etSTED slow method scan. """
        self.setDetLogLine("scan_end",datetime.now().strftime('%Ss%fus'))
        # TODO: emit signal to save the last scanned image #xxx.sigSnapImg.emit()
        self.endRecording()
        self.continueFastModality()
        self.__fast_frame = 0

    def setDetLogLine(self, key, val, *args):
        if args:
            self.__detLog[f"{key}{args[0]}"] = val
        else:
            self.__detLog[key] = val

    def runSlowScan(self):
        """ Run event-triggered scan in small ROI. """
        print(self._scanParameterDict)
        # TODO: emit signal to run scan #xxx.sigRunScan.emit(self.signalDict)

    def endRecording(self):
        """ Save an etSTED slow method scan. """
        self.setDetLogLine("pipeline", self.getPipelineName())
        self.logPipelineParamVals()
        # save log file with temporal info of trigger event
        filename = datetime.utcnow().strftime('%Hh%Mm%Ss%fus')
        name = os.path.join(_logsDir, filename) + '_log'
        savename = getUniqueName(name)
        log = [f'{key}: {self.__detLog[key]}' for key in self.__detLog]
        with open(f'{savename}.txt', 'w') as f:
            [f.write(f'{st}\n') for st in log]
        self.resetDetLog()

    def getTransformName(self):
        """ Get the name of the pipeline currently used. """
        transformidx = self._widget.transformPipelinePar.currentIndex()
        transformname = self._widget.transformPipelines[transformidx]
        return transformname

    def getPipelineName(self):
        """ Get the name of the pipeline currently used. """
        pipelineidx = self._widget.analysisPipelinePar.currentIndex()
        pipelinename = self._widget.analysisPipelines[pipelineidx]
        return pipelinename

    def logPipelineParamVals(self):
        """ Put analysis pipeline parameter values in the log file. """
        params_ignore = ['img','bkg','binary_mask','testmode','exinfo']
        param_names = list()
        for pipeline_param_name, _ in self.__pipeline_params.items():
            if pipeline_param_name not in params_ignore:
                param_names.append(pipeline_param_name)
        for key, val in zip(param_names, self.__pipeline_param_vals):
            self.setDetLogLine(key, val)

    def continueFastModality(self):
        """ Continue the fast method, after an event scan has been performed. """
        if self._widget.endlessScanCheck.isChecked() and not self.__running:
            # connect communication channel signals
            # TODO: connect signal from update of image to running pipeline #xxx.sigUpdateImage.connect(self.runPipeline)
            # TODO: turn on laserFast #xxx.lasersManager.laserFast.setEnabled(True)
            self._widget.initiateButton.setText('Stop')
            self.__running = True
        elif not self._widget.endlessScanCheck.isChecked():
            # TODO: disconnect signal from end of scan to scanEnded() #xxx.sigScanEnded.disconnect(self.scanEnded)
            self._widget.initiateButton.setText('Initiate')
            self.__running = False
            self.resetPipelineParamVals()

    def loadTransform(self):
        """ Load a previously saved coordinate transform. """
        transformname = self.getTransformName()
        self.transform = getattr(importlib.import_module(f'{transformname}'), f'{transformname}')

    def loadPipeline(self):
        """ Load the selected analysis pipeline, and its parameters into the GUI. """
        pipelinename = self.getPipelineName()
        self.pipeline = getattr(importlib.import_module(f'{pipelinename}'), f'{pipelinename}')
        self.__pipeline_params = signature(self.pipeline).parameters
        self._widget.initParamFields(self.__pipeline_params, self.__params_exclude)

    def initiateBinaryMask(self):
        """ Initiate the process of calculating a binary mask of the region of interest. """
        self.__binary_stack = None
        # TODO: turn on laserFast #xxx.lasersManager.laserFast.setEnabled(True)
        # TODO: connect signal from update of image to saving the image in the stack of images for binary mask calculation
        self.camImgWorker.newFrame.connect(self.addImgBinStack)
        self._widget.recordBinaryMaskButton.setText('Recording...')

    def addImgBinStack(self, img):
        """ Add image to the stack of images used to calculate a binary mask of the region of interest. """
        if self.__binary_stack is None:
            self.__binary_stack = img
        elif len(self.__binary_stack) == self.__binary_frames:
            # TODO: disconnect signal from update of image to saving the image in the stack of images for binary mask calculation
            self.camImgWorker.newFrame.disconnect(self.addImgBinStack)
            # TODO: turn off laserFast #xxx.lasersManager.laserFast.setEnabled(False)
            self.calculateBinaryMask(self.__binary_stack)
        else:
            if np.ndim(self.__binary_stack) == 2:
                self.__binary_stack = np.stack((self.__binary_stack, img))
            else:
                self.__binary_stack = np.concatenate((self.__binary_stack,  [img]), axis=0)

    def calculateBinaryMask(self, img_stack):
        """ Calculate the binary mask of the region of interest. """
        img_mean = np.mean(img_stack, 0)
        img_bin = ndi.filters.gaussian_filter(img_mean, np.float(self._widget.bin_smooth_edit.text()))
        self.__binary_mask = np.array(img_bin > np.float(self._widget.bin_thresh_edit.text()))
        self._widget.recordBinaryMaskButton.setText('Record binary mask')
        self.setAnalysisHelpImg(self.__binary_mask)
        self.launchHelpWidget()

    def setAnalysisHelpImg(self, img):
        """ Set the preprocessed image in the analysis help widget. """
        if self.__fast_frame < self.__init_frames + 3:
            autolevels = True
        else:
            autolevels = False
        self._widget.analysisHelpWidget.img.setImage(img, autoLevels=autolevels)
        infotext = f'Min: {np.min(img)}, max: {np.max(img/10000)} (rel. change)'
        self._widget.analysisHelpWidget.info_label.setText(infotext)

    def getScanParameters(self):
        """ Get scan parameters (size (per axis), pixel size (per axis), dwell time etc) from a scanning widget/scan part of software. """
        self._scanParameterDict = {
            'target_device': ['X-galvo', 'Y-galvo'],
            'axis_size': [5,5],
            'axis_centerpos': [0,0],
            'axis_pixel_size': [0.03, 0.03],
            'dwell_time': 0.03
        }

    def setBusyFalse(self):
        """ Set busy flag to false. """
        self.__busy = False

    def readPipelineParams(self):
        """ Read user-provided analysis pipeline parameter values. """
        param_vals = list()
        for item in self._widget.param_edits:
            param_vals.append(np.float(item.text()))
        return param_vals

    def launchHelpWidget(self):
        """ Launch help widget that shows the preprocessed images in real-time. """
        self._widget.launchHelpWidget(self._widget.analysisHelpWidget, init=True)

    def resetDetLog(self):
        """ Reset the event log dictionary. """
        self.__detLog = dict()

    def resetPipelineParamVals(self):
        """ Reset the pipeline parameters. """
        self.__pipeline_param_vals = list()

    def resetRunParams(self):
        """ Reset general pipeline run parameters. """
        self.__running = False
        self.__validating = False
        self.__fast_frame = 0
        self.__post_event_frames = 0

    def runPipeline(self, img):
        """ Run the analyis pipeline, called after every fast method frame. """
        if not self.__busy:
            # if not still running pipeline on last frame
            self.__busy = True
            # log start of pipeline
            self.setDetLogLine("pipeline_start", datetime.now().strftime('%Ss%fus'))

            # run pipeline
            if self.__runMode == RunMode.TestVisualize or self.__runMode == RunMode.TestValidate:
                # if chosen a test mode: run pipeline with analysis image return
                coords_detected, self.__exinfo, img_ana = self.pipeline(img, self.__bkg, self.__binary_mask,
                                                                        (self.__runMode==RunMode.TestVisualize or
                                                                        self.__runMode==RunMode.TestValidate),
                                                                        self.__exinfo, *self.__pipeline_param_vals)
            else:
                # if chosen experiment mode: run pipeline without analysis image return
                coords_detected, self.__exinfo = self.pipeline(img, self.__bkg, self.__binary_mask,
                                                               self.__runMode==RunMode.TestVisualize,
                                                               self.__exinfo, *self.__pipeline_param_vals)
            self.setDetLogLine("pipeline_end", datetime.now().strftime('%Ss%fus'))

            if self.__fast_frame > self.__init_frames:
                # if initial settling frames have passed
                if self.__runMode == RunMode.TestVisualize:
                    # if visualization mode: set analysis image in help widget
                    self.setAnalysisHelpImg(img_ana)
                elif self.__runMode == RunMode.TestValidate:
                    # if validation mode: set analysis image in help widget,
                    # and start to record validation frames after event
                    self.setAnalysisHelpImg(img_ana)
                    if self.__validating:
                        # if currently validating
                        if self.__post_event_frames > self.__validation_frames:
                            # if all validation frames have been recorded, pause fast imaging,
                            # end recording, and then continue fast imaging
                            self.saveValidationImages(prev=True, prev_ana=True)
                            self.pauseFastModality()
                            self.endRecording()
                            self.continueFastModality()
                            self.__fast_frame = 0
                            self.__validating = False
                        self.__post_event_frames += 1
                    elif coords_detected.size != 0:
                        # if some events where detected and not validating
                        # take first detected coords as event
                        if np.size(coords_detected) > 2:
                            coords_scan = coords_detected[0,:]
                        else:
                            coords_scan = coords_detected[0]
                        # log detected center coordinate
                        self.setDetLogLine("fastscan_x_center", coords_scan[0])
                        self.setDetLogLine("fastscan_y_center", coords_scan[1])
                        # flag for start of validation
                        self.__validating = True
                        self.__post_event_frames = 0
                elif coords_detected.size != 0:
                    # if experiment mode, and some events were detected
                    # take first detected coords as event
                    if np.size(coords_detected) > 2:
                        coords_scan = coords_detected[0,:]
                    else:
                        coords_scan = coords_detected[0]
                    self.setDetLogLine("prepause", datetime.now().strftime('%Ss%fus'))
                    # pause fast imaging
                    self.pauseFastModality()
                    self.setDetLogLine("coord_transf_start", datetime.now().strftime('%Ss%fus'))
                    # transform detected coordinate between fast and scanning imaging spaces
                    coords_center_scan = self.transform(coords_scan, self.__transformCoeffs)
                    # log detected and scanning center coordinate
                    self.setDetLogLine("fastscan_x_center", coords_scan[0])
                    self.setDetLogLine("fastscan_y_center", coords_scan[1])
                    self.setDetLogLine("slowscan_x_center", coords_center_scan[0])
                    self.setDetLogLine("slowscan_y_center", coords_center_scan[1])
                    self.setDetLogLine("scan_initiate", datetime.now().strftime('%Ss%fus'))
                    # initiate and run scanning with transformed center coordinate
                    self.initiateSlowScan(position=coords_center_scan)
                    self.runSlowScan()

                    # buffer latest fast frame and save validation images
                    self.__prevFrames.append(img)
                    self.saveValidationImages(prev=True, prev_ana=False)
                    self.__busy = False
                    return
            # use latest fast frame as background for next pipeline run
            self.__bkg = img
            # buffer latest fast frame and save validation images
            self.__prevFrames.append(img)
            if self.__runMode == RunMode.TestValidate:
                # if validation mode: buffer previous preprocessed analysis frame
                self.__prevAnaFrames.append(img_ana)
            self.__fast_frame += 1
            # unset busy flag
            self.setBusyFalse()


    def initiateSlowScan(self, position=[0.0,0.0]):
        """ Initiate a STED scan. """
        # change the center coordinate of the scan parameters to the detected positions
        self.setCenterScanParameter(position)
        # generate scanning curves through scanning part of software and save to self.signalDic
        # TODO: self.signalDict = xxx.genereateScanCurves(self._scanParameterDict)
        self.signalDict = {}  # mock: empty signal dictionaries

    def setCenterScanParameter(self, position):
        """ Set the scanning center from the detected event coordinates. """
        if self._scanParameterDict != {}:
            # if scan parameters have been loaded
            self._scanParameterDict['axis_centerpos'] = []
            # null center positions
            for index,_ in enumerate(self._scanParameterDict['target_device']):
                # for each scanning device (assuming X fast and Y slow)
                center = position[index]
                if index==0:
                    # if fast axis: add shift to detected position due to scanning lag etc
                    center = self.addFastAxisShift(center)
                # save event coordinate as center for scanning device
                self._scanParameterDict['axis_centerpos'].append(center)

    def addFastAxisShift(self, center):
        """ Add a scanning-method and microscope-specific shift to the fast axis scanning. 
        For Alvelid et al 2022: based on second-degree curved surface fit to 2D-sampling
        of dwell time and pixel size induced shifts. """
        dwell_time = float(self._scanParameterDict['dwell_time'])
        px_size = float(self._scanParameterDict['axis_pixel_size'][0])
        C = np.array([0, 0, 0, 0, 0, 0])  # second order plane fit, here mock null shift
        params = np.array([px_size**2, dwell_time**2, px_size*dwell_time, px_size, dwell_time, 1])  # for use with second order plane fit
        shift_compensation = np.sum(params*C)
        center -= shift_compensation
        return(center)

    def saveValidationImages(self, prev=True, prev_ana=True):
        """ Save the validation fast images of an event detection, fast images and/or preprocessed analysis images. """
        if prev:
            # TODO: save detectorFast frames leading up to event #xxx.sigSaveImage.emit(self.detectorFast, np.array(list(self.__prevFrames)), 'raw') # (detector, imagestack, name_suffix)
            self.__prevFrames.clear()
        if prev_ana:
            # TODO: save preprocessed frames leading up to event #xxx.sigSaveImage.emit(self.detectorFast, np.array(list(self.__prevAnaFrames)), 'ana') # (detector, imagestack, name_suffix)
            self.__prevAnaFrames.clear()

    def pauseFastModality(self):
        """ Pause the fast method, when an event has been detected. """
        if self.__running:
            # TODO: disconnect signal from update of image to running pipeline #xxx.sigUpdateImage.disconnect(self.runPipeline)
            self.camImgWorker.newFrame.disconnect(self.runPipeline)  # mock: directly from mock camera worker
            # TODO: turn off fast laser xxx.lasersManager.laserFast.setEnabled(False)
            self.__running = False

    def closeEvent(self):
        pass


class EtSTEDCoordTransformHelper():
    """ Coordinate transform help widget controller. """
    def __init__(self, etSTEDController, coordTransformWidget, saveFolder, *args, **kwargs):

        self.etSTEDController = etSTEDController
        self._widget = coordTransformWidget
        self.__saveFolder = saveFolder

        # initiate coordinate transform parameters
        self.__transformCoeffs = [0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0]  # unit transformation
        self.__loResCoords = list()
        self.__hiResCoords = list()
        self.__loResCoordsPx = list()
        self.__hiResCoordsPx = list()
        self.__hiResPxSize = 1
        self.__loResPxSize = 1
        self.__hiResSize = 1

        # connect signals from widget
        etSTEDController._widget.coordTransfCalibButton.clicked.connect(self.calibrationLaunch)
        self._widget.saveCalibButton.clicked.connect(self.calibrationFinish)
        self._widget.resetCoordsButton.clicked.connect(self.resetCalibrationCoords)
        self._widget.loadLoResButton.clicked.connect(lambda: self.loadCalibImage('lo'))
        self._widget.loadHiResButton.clicked.connect(lambda: self.loadCalibImage('hi'))

    def getTransformCoeffs(self):
        """ Get transformation coefficients. """
        return self.__transformCoeffs

    def calibrationLaunch(self):
        """ Launch calibration. """
        self.etSTEDController._widget.launchHelpWidget(self.etSTEDController._widget.coordTransformWidget, init=True)

    def calibrationFinish(self):
        """ Finish calibration. """
        # get annotated coordinates in both images and translate to real space coordinates
        self.__loResCoordsPx = self._widget.pointsLayerLo.data
        for pos_px in self.__loResCoordsPx:
            pos = (np.around(pos_px[0]*self.__loResPxSize, 3), np.around(pos_px[1]*self.__loResPxSize, 3))
            self.__loResCoords.append(pos)
        self.__hiResCoordsPx = self._widget.pointsLayerHi.data
        for pos_px in self.__hiResCoordsPx:
            pos = (np.around(pos_px[0]*self.__hiResPxSize - self.__hiResSize/2, 3), -1 * np.around(pos_px[1]*self.__hiResPxSize - self.__hiResSize/2, 3))
            self.__hiResCoords.append(pos)
        # calibrate coordinate transform
        self.coordinateTransformCalibrate()
        print(f'Transformation coeffs: {self.__transformCoeffs}')
        name = datetime.utcnow().strftime('%Hh%Mm%Ss%fus')
        filename = os.path.join(self.__saveFolder, name) + '_transformCoeffs.txt'
        np.savetxt(fname=filename, X=self.__transformCoeffs)

        # plot the resulting transformed low-res coordinates on the hi-res image
        coords_transf = []
        for i in range(0,len(self.__loResCoords)):
            pos = self.poly_thirdorder_transform(self.__transformCoeffs, self.__loResCoords[i])
            pos_px = (np.around((pos[0] + self.__hiResSize/2)/self.__hiResPxSize, 0), np.around((-1 * pos[1] + self.__hiResSize/2)/self.__hiResPxSize, 0))
            coords_transf.append(pos_px)
        coords_transf = np.array(coords_transf)
        self._widget.pointsLayerTransf.data = coords_transf

    def resetCalibrationCoords(self):
        """ Reset all selected coordinates. """
        self.__loResCoords = list()
        self.__loResCoordsPx = list()
        self.__hiResCoords = list()
        self.__hiResCoordsPx = list()
        self._widget.pointsLayerLo.data = []
        self._widget.pointsLayerHi.data = []
        self._widget.pointsLayerTransf.data = []

    def loadCalibImage(self, modality):
        """ Load low or high resolution calibration image. """
        # open gui to choose file
        img_filename = self.openFolder()
        # load img data from file
        with h5py.File(img_filename, "r") as f:
            img_key = list(f.keys())[0]
            pixelsize = f.attrs['element_size_um'][1]
            print(pixelsize)
            img_data = np.array(f[img_key])
            imgsize = pixelsize*np.size(img_data,0)
        # view data in corresponding viewbox
        self.updateCalibImage(img_data, modality)
        if modality == 'hi':
            self.__hiResCoords = list()
            self.__hiResPxSize = pixelsize
            self.__hiResSize = imgsize
        elif modality == 'lo':
            self.__loResCoords = list()
            self.__loResPxSize = pixelsize

    def findFile(self):
        """ Opens current folder in the file explorer and returns chosen filename. """
        Tk().withdraw()
        filename = filedialog.askopenfilename()
        return filename

    def updateCalibImage(self, img_data, modality):
        """ Update new image in the viewbox. """
        if modality == 'hi':
            viewer = self._widget.napariViewerHi
        elif modality == 'lo':
            viewer = self._widget.napariViewerLo
        viewer.add_image(img_data)
        viewer.layers.unselect_all()
        viewer.layers.move_selected(len(viewer.layers)-1,0)

    def coordinateTransformCalibrate(self):
        """ Third-order polynomial fitting with least-squares Levenberg-Marquart algorithm. """
        # prepare data and init guess
        c_init = np.hstack([np.zeros(10), np.zeros(10)])
        xdata = np.array([*self.__loResCoords]).astype(np.float32)
        ydata = np.array([*self.__hiResCoords]).astype(np.float32)
        initguess = c_init.astype(np.float32)
        # fit
        res_lsq = least_squares(self.poly_thirdorder, initguess, args=(xdata, ydata), method='lm')
        transformCoeffs = res_lsq.x
        self.__transformCoeffs = transformCoeffs

    def poly_thirdorder(self, a, x, y):
        """ Polynomial function that will be fit in the least-squares fit. """
        res = []
        for i in range(0, len(x)):
            c1 = x[i,0]
            c2 = x[i,1]
            x_i1 = a[0]*c1**3 + a[1]*c2**3 + a[2]*c2*c1**2 + a[3]*c1*c2**2 + a[4]*c1**2 + a[5]*c2**2 + a[6]*c1*c2 + a[7]*c1 + a[8]*c2 + a[9]
            x_i2 = a[10]*c1**3 + a[11]*c2**3 + a[12]*c2*c1**2 + a[13]*c1*c2**2 + a[14]*c1**2 + a[15]*c2**2 + a[16]*c1*c2 + a[17]*c1 + a[18]*c2 + a[19]
            res.append(x_i1 - y[i,0])
            res.append(x_i2 - y[i,1])
        return res
    
    def poly_thirdorder_transform(self, a, x):
        """ Use for plotting the least-squares fit results. """
        c1 = x[0]
        c2 = x[1]
        x_i1 = a[0]*c1**3 + a[1]*c2**3 + a[2]*c2*c1**2 + a[3]*c1*c2**2 + a[4]*c1**2 + a[5]*c2**2 + a[6]*c1*c2 + a[7]*c1 + a[8]*c2 + a[9]
        x_i2 = a[10]*c1**3 + a[11]*c2**3 + a[12]*c2*c1**2 + a[13]*c1*c2**2 + a[14]*c1**2 + a[15]*c2**2 + a[16]*c1*c2 + a[17]*c1 + a[18]*c2 + a[19]
        return (x_i1, x_i2)


class RunMode(enum.Enum):
    Experiment = 1
    TestVisualize = 2
    TestValidate = 3

def insertSuffix(filename, suffix, newExt=None):
    names = os.path.splitext(filename)
    if newExt is None:
        return names[0] + suffix + names[1]
    else:
        return names[0] + suffix + newExt

def getUniqueName(name):
    name, ext = os.path.splitext(name)
    n = 1
    while glob.glob(name + ".*"):
        if n > 1:
            name = name.replace('_{}'.format(n - 1), '_{}'.format(n))
        else:
            name = insertSuffix(name, '_{}'.format(n))
        n += 1
    return ''.join((name, ext))


# Copyright (C) 2020-2022 ImSwitch developers
#
# ImSwitch is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ImSwitch is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
