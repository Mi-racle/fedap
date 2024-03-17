import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QTableWidget, QPushButton, QWidget, QTableWidgetItem, QFileDialog, QProgressBar


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("表格示例")
        self.setGeometry(100, 100, 800, 600)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.layout = QVBoxLayout()

        self.table = QTableWidget()
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(["节点名", "目标服务器IP", "数据集", "控制", "训练进度"])

        self.add_row_button = QPushButton("添加节点")
        self.add_row_button.clicked.connect(self.add_row)

        self.layout.addWidget(self.table)
        self.layout.addWidget(self.add_row_button)

        self.central_widget.setLayout(self.layout)

    def add_row(self):
        row_position = self.table.rowCount()
        self.table.insertRow(row_position)

        # 添加字符串项
        string_item = QTableWidgetItem("节点名")
        self.table.setItem(row_position, 0, string_item)

        # 添加IP地址项
        ip_item = QTableWidgetItem("127.0.0.1")
        self.table.setItem(row_position, 1, ip_item)

        # 添加选择文件按钮
        select_file_button = QPushButton("选择文件夹")
        select_file_button.clicked.connect(lambda _, row=row_position: self.select_file(row))
        self.table.setCellWidget(row_position, 2, select_file_button)

        # 添加按钮
        button = QPushButton("开始")
        button.clicked.connect(lambda _, btn=button: self.toggle_button(btn))
        self.table.setCellWidget(row_position, 3, button)

        # 添加进度条
        progress_bar = QProgressBar()
        self.table.setCellWidget(row_position, 4, progress_bar)

    def select_file(self, row):
        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.FileMode.ExistingFile)
        if file_dialog.exec():
            file_path = file_dialog.selectedFiles()[0]
            self.table.item(row, 2).setText(file_path)

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
