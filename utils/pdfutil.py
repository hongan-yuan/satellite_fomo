# -*- coding: utf-8 -*-
"""
@Time : 
@Author: Honggang Yuan
@Email: hn_yuanhg@163.com
Description:
    
"""
from PIL import Image
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter


def bmp_to_pdf(bmp_file_path, pdf_file_path):
    # 打开BMP文件
    image = Image.open(bmp_file_path)

    # 获取图像的宽度和高度
    width, height = image.size
    width = width    # 创建一个PDF文件
    c = canvas.Canvas(pdf_file_path, pagesize=(width, height))

    # 将图像绘制到PDF文件中
    c.drawImage(bmp_file_path, 0, 0, width, height)

    # 保存PDF文件
    c.save()


# 使用示例
# bmp_file_path = 'energy_distribution.bmp'
# pdf_file_path = 'energy_distribution.pdf'
bmp_file_path = 'constellation48.bmp'
pdf_file_path = 'constellation48.pdf'
bmp_to_pdf(bmp_file_path, pdf_file_path)
