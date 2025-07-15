from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.utils import ImageReader
from datetime import datetime
import os

font_path = os.path.join("fonts", "DejaVuSans.ttf")
pdfmetrics.registerFont(TTFont("DejaVu", font_path))

def generate_pdf_report(input_img_path, result_img_path, stats, report_path):
    c = canvas.Canvas(report_path, pagesize=letter)
    width, height = letter

    c.setFont("DejaVu", 16)
    c.drawString(50, height - 50, "Отчёт по обработке изображения парковки")

    c.setFont("DejaVu", 12)
    c.drawString(50, height - 80, f"Дата: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    c.drawString(50, height - 100, f"Статистика:")

    y = height - 120
    for key, val in stats.items():
        c.drawString(70, y, f"{key}: {val}")
        y -= 20

    c.drawString(50, y-10, "Исходное изображение:")
    input_img = ImageReader(input_img_path)
    c.drawImage(input_img, 50, y - 210, width=250, height=200)

    c.drawString(320, y-10, "Результат обработки:")
    result_img = ImageReader(result_img_path)
    c.drawImage(result_img, 320, y - 210, width=250, height=200)

    c.showPage()
    c.save()
