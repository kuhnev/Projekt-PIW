from app import App
from PyQt5.QtWidgets import QApplication
import sys

def main():
    app = QApplication(sys.argv)
    ex = App("./dataset/dataset.avi")
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
