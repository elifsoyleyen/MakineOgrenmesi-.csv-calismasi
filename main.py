# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 21:03:07 2020

@author: elif
"""

import sys
from PyQt5 import QtCore, QtGui, QtWidgets
from kod import MainWindow

def main():
    app = QtWidgets.QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
    



