import random
import sys
import threading
import time

import flwr as fl
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QTableWidget, QPushButton, QWidget, \
    QTableWidgetItem, QFileDialog, QHBoxLayout, QTabWidget, QGridLayout, QLabel, QTextEdit, QFrame
from datasets import load_dataset, load_from_disk

from clients.fedclient import FedClient
from testui import Ale
from utils import apply_transforms


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle('锁销分类系统')
        self.setGeometry(100, 100, 800, 600)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.layout = QVBoxLayout()

        self.tab_widget = QTabWidget()
        self.layout.addWidget(self.tab_widget)

        self.tab1 = QWidget()
        self.tab_widget.addTab(self.tab1, '联邦学习')
        self.tab1.layout = QVBoxLayout()

        table_head = ['节点名', '目标服务器IP', '数据集', '操作', '状态']
        self.table = QTableWidget()
        self.table.setColumnCount(len(table_head))
        self.table.setColumnWidth(3, 300)
        self.table.setHorizontalHeaderLabels(table_head)

        self.add_row_button = QPushButton("添加节点")
        self.add_row_button.clicked.connect(self.add_row)

        self.node_holder = 1

        self.tab1.layout.addWidget(self.table)
        self.tab1.layout.addWidget(self.add_row_button)
        self.tab1.setLayout(self.tab1.layout)

        self.tab2 = QWidget()
        self.tab_widget.addTab(self.tab2, '锁销分类')
        self.tab2.layout = QGridLayout()

        self.image_label = QLabel('选择图片后将在此显示')
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)  # 设置文本左上角对齐
        self.image_label.setStyleSheet('background-color: white')
        self.image_label.setFixedSize(300, 300)
        self.tab2.layout.addWidget(self.image_label, 0, 0)

        self.result_text = QTextEdit('分类结果将在此显示')
        self.result_text.setReadOnly(True)
        self.result_text.setStyleSheet('border: none;')
        self.result_text.setFixedSize(300, 300)
        self.tab2.layout.addWidget(self.result_text, 0, 1)

        self.left_widget = QWidget()
        self.left_widget.setFixedHeight(40)
        self.left_widget.layout = QHBoxLayout()
        select_image_button = QPushButton('选择图片')
        select_image_button.clicked.connect(self.select_image)
        self.left_widget.layout.addWidget(select_image_button)
        self.left_widget.setLayout(self.left_widget.layout)
        self.tab2.layout.addWidget(self.left_widget, 1, 0)

        self.right_widget = QWidget()
        self.right_widget.setFixedHeight(40)
        self.right_widget.layout = QHBoxLayout()
        select_model_button = QPushButton('选择模型')
        select_model_button.clicked.connect(self.select_model)
        self.right_widget.layout.addWidget(select_model_button)
        classify_button = QPushButton('开始分类')
        classify_button.clicked.connect(self.classify_image)
        self.right_widget.layout.addWidget(classify_button)
        self.model_path = ''
        self.right_widget.setLayout(self.right_widget.layout)
        self.tab2.layout.addWidget(self.right_widget, 1, 1)

        self.tab2.setLayout(self.tab2.layout)

        self.central_widget.setLayout(self.layout)

    def add_row(self):
        row_position = self.table.rowCount()
        self.table.insertRow(row_position)

        string_item = QTableWidgetItem(f'节点名{self.node_holder}')
        self.table.setItem(row_position, 0, string_item)

        # ip_item = QTableWidgetItem('192.168.1.4')
        ip_item = QTableWidgetItem('localhost:8080')
        self.table.setItem(row_position, 1, ip_item)

        folder_item = QTableWidgetItem('尚未选择')
        folder_item.setSizeHint(folder_item.sizeHint())
        folder_item.setFlags(folder_item.flags() ^ Qt.ItemFlag.ItemIsEditable)
        self.table.setItem(row_position, 2, folder_item)

        button_widget = QWidget()
        layout = QHBoxLayout()

        select_folder_button = QPushButton("选择数据集")
        select_folder_button.clicked.connect(lambda _, row=row_position: self.select_folder(row))
        layout.addWidget(select_folder_button)

        start_stop_button = QPushButton("加入联邦")
        start_stop_button.clicked.connect(
            lambda _, row=row_position, btn=start_stop_button: self.federate_button(row, btn))
        layout.addWidget(start_stop_button)

        delete_stop_button = QPushButton("删除节点")
        delete_stop_button.clicked.connect(lambda _, row=row_position: self.delete_button(row))
        layout.addWidget(delete_stop_button)

        delete_stop_button = QPushButton("导出模型")
        delete_stop_button.clicked.connect(lambda _, row=row_position: self.export_button(row))
        layout.addWidget(delete_stop_button)

        button_widget.setLayout(layout)

        self.table.setCellWidget(row_position, 3, button_widget)

        status_item = QTableWidgetItem('闲置')
        status_item.setFlags(status_item.flags() ^ Qt.ItemFlag.ItemIsEditable)
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

    def federate_button(self, row, button):
        if button.text() == "加入联邦":
            data_path = self.table.item(row, 2).text()
            trainset = load_from_disk(f'{data_path}/train')
            validset = load_from_disk(f'{data_path}/valid')
            trainset = trainset.with_transform(apply_transforms)
            validset = validset.with_transform(apply_transforms)

            thread = threading.Thread(
                target=fl.client.start_numpy_client(
                    server_address="localhost:8080",
                    client=FedClient(trainset, validset, int(row))
                )
            )
            thread.start()

            status_item = QTableWidgetItem('训练中')
            status_item.setForeground(Qt.GlobalColor.green)
            self.table.setItem(row, 4, status_item)
            button.setText("退出联邦")
        else:
            status_item = QTableWidgetItem('闲置')
            self.table.setItem(row, 4, status_item)
            button.setText("加入联邦")

    def delete_button(self, row):
        self.table.removeRow(row)

        for row in range(self.table.rowCount()):
            widget = self.table.cellWidget(row, 3)
            buttons = widget.findChildren(QPushButton)

            buttons[0].clicked.disconnect()
            buttons[0].clicked.connect(lambda _, _row=row: self.select_folder(_row))

            buttons[2].clicked.disconnect()
            buttons[2].clicked.connect(lambda _, _row=row: self.delete_button(_row))

    def export_button(self, row):
        pass

    def select_image(self):
        file_dialog = QFileDialog(self)
        file_dialog.setWindowTitle('选择图片')
        file_dialog.setFileMode(QFileDialog.FileMode.ExistingFile)
        file_dialog.setNameFilter('Images (*.png *.jpg *.jpeg *.bmp)')
        if file_dialog.exec():
            file_path = file_dialog.selectedFiles()[0]
            pixmap = QPixmap(file_path)
            self.image_label.setPixmap(pixmap.scaled(300, 300, aspectRatioMode=Qt.AspectRatioMode.IgnoreAspectRatio))
            # self.image_label.setPixmap(pixmap.scaled(300, 300, aspectRatioMode=Qt.AspectRatioMode.KeepAspectRatio))

    def select_model(self):
        file_dialog = QFileDialog(self)
        file_dialog.setWindowTitle('选择模型')
        file_dialog.setFileMode(QFileDialog.FileMode.ExistingFile)
        file_dialog.setNameFilter('(*.pt)')
        if file_dialog.exec():
            self.model_path = file_dialog.selectedFiles()[0]

    def classify_image(self):
        conf = random.random()
        self.result_text.setText(
            '类型：A01\n' +
            f'置信度：{conf}'
        )


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
