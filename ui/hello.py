import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QTableWidget, QPushButton, QWidget, \
    QTableWidgetItem, QFileDialog, QProgressBar, QHBoxLayout


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("锁销分类联邦学习网络")
        self.setGeometry(100, 100, 800, 600)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.layout = QVBoxLayout()

        table_head = ["节点名", "目标服务器IP", "数据集", "操作", "状态"]
        self.table = QTableWidget()
        self.table.setColumnCount(len(table_head))
        self.table.setColumnWidth(3, 240)
        self.table.setHorizontalHeaderLabels(table_head)

        self.add_row_button = QPushButton("添加节点")
        self.add_row_button.clicked.connect(self.add_row)

        self.layout.addWidget(self.table)
        self.layout.addWidget(self.add_row_button)

        self.central_widget.setLayout(self.layout)

        self.node_holder = 1

    def add_row(self):
        row_position = self.table.rowCount()
        self.table.insertRow(row_position)

        string_item = QTableWidgetItem(f'节点名{self.node_holder}')
        self.table.setItem(row_position, 0, string_item)

        ip_item = QTableWidgetItem('127.0.0.1')
        self.table.setItem(row_position, 1, ip_item)

        folder_item = QTableWidgetItem('尚未选择')
        folder_item.setSizeHint(folder_item.sizeHint())
        self.table.setItem(row_position, 2, folder_item)

        button_widget = QWidget()
        layout = QHBoxLayout()

        select_folder_button = QPushButton("选择文件夹")
        select_folder_button.clicked.connect(lambda _, row=row_position: self.select_folder(row))
        layout.addWidget(select_folder_button)

        start_stop_button = QPushButton("加入联邦")
        start_stop_button.clicked.connect(lambda _, btn=start_stop_button: self.federate_button(btn))
        layout.addWidget(start_stop_button)

        delete_stop_button = QPushButton("删除节点")
        delete_stop_button.clicked.connect(lambda _, row=row_position: self.delete_button(row))
        layout.addWidget(delete_stop_button)

        button_widget.setLayout(layout)

        self.table.setCellWidget(row_position, 3, button_widget)

        status_item = QTableWidgetItem('闲置')
        self.table.setItem(row_position, 4, status_item)

        self.table.setRowHeight(row_position, 60)

        self.node_holder += 1

    def select_folder(self, row):
        folder_dialog = QFileDialog()
        folder_dialog.setFileMode(QFileDialog.FileMode.Directory)
        if folder_dialog.exec():
            folder_path = folder_dialog.selectedFiles()[0]
            folder_item = QTableWidgetItem(folder_path)
            self.table.setItem(row, 2, folder_item)

    def federate_button(self, button):
        if button.text() == "加入联邦":
            button.setText("退出联邦")
        else:
            button.setText("加入联邦")

    def delete_button(self, row):
        self.table.removeRow(row)

        for row in range(self.table.rowCount()):
            widget = self.table.cellWidget(row, 3)
            buttons = widget.findChildren(QPushButton)

            buttons[0].clicked.disconnect()
            buttons[0].clicked.connect(lambda _, row=row: self.select_folder(row))

            buttons[2].clicked.disconnect()
            buttons[2].clicked.connect(lambda _, row=row: self.delete_button(row))

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
