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

        self.table = QTableWidget()
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(["节点名", "目标服务器IP", "数据集", "操作", "训练进度"])

        self.add_row_button = QPushButton("添加节点")
        self.add_row_button.clicked.connect(self.add_row)

        self.layout.addWidget(self.table)
        self.layout.addWidget(self.add_row_button)

        self.central_widget.setLayout(self.layout)

    def add_row(self):
        row_position = self.table.rowCount()
        self.table.insertRow(row_position)

        # 添加字符串项
        string_item = QTableWidgetItem('节点名')
        self.table.setItem(row_position, 0, string_item)

        # 添加IP地址项
        ip_item = QTableWidgetItem('127.0.0.1')
        self.table.setItem(row_position, 1, ip_item)

        # 添加IP地址项
        folder_item = QTableWidgetItem('尚未选择')
        folder_item.setSizeHint(folder_item.sizeHint())
        self.table.setItem(row_position, 2, folder_item)

        # 创建一个QWidget，包含两个按钮
        button_widget = QWidget()
        layout = QHBoxLayout()

        # 添加选择文件夹按钮
        select_folder_button = QPushButton("选择文件夹")
        select_folder_button.clicked.connect(lambda _, row=row_position: self.select_folder(row))
        layout.addWidget(select_folder_button)

        # 添加开始/结束按钮
        start_stop_button = QPushButton("开始")
        start_stop_button.clicked.connect(lambda _, btn=start_stop_button: self.toggle_button(btn))
        layout.addWidget(start_stop_button)

        button_widget.setLayout(layout)

        # 将QWidget设置为表格的单元格
        self.table.setCellWidget(row_position, 3, button_widget)

        # 添加进度条
        progress_bar = QProgressBar()
        self.table.setCellWidget(row_position, 4, progress_bar)

    def select_folder(self, row):
        folder_dialog = QFileDialog()
        folder_dialog.setFileMode(QFileDialog.FileMode.Directory)
        if folder_dialog.exec():
            folder_path = folder_dialog.selectedFiles()[0]
            folder_item = QTableWidgetItem(folder_path)
            self.table.setItem(row, 2, folder_item)

    def toggle_button(self, button):
        if button.text() == "开始":
            button.setText("结束")
        else:
            button.setText("开始")


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
