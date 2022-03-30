from EtSTEDController import EtSTEDController
from EtSTEDWidget import EtSTEDWidget
from qtpy import QtWidgets
import sys

etSTEDapp = QtWidgets.QApplication(sys.argv)
widget = EtSTEDWidget()
controller = EtSTEDController(widget)

widget.show()
sys.exit(etSTEDapp.exec_())
