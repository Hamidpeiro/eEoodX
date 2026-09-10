import sys
import json
import asyncio
import threading

from PySide6.QtWidgets import QApplication, QWidget
from PySide6.QtGui import QPainter, QPen, QBrush
from PySide6.QtCore import Qt, QPointF

import websockets


HOST = "127.0.0.1"
PORT = 8765


class Viewer(QWidget):

    def __init__(self):
        super().__init__()

        self.data = {
            "outline": [],
            "cracks": [],
            "knots": [],
            "cuts": []
        }

        self.setWindowTitle("eWoodX Viewer")
        self.resize(1200, 800)

    def update_data(self, data):
        self.data = data
        self.update()

    def paintEvent(self, event):

        painter = QPainter(self)

        # Black background
        painter.fillRect(self.rect(), Qt.black)

        # -------------------------
        # OUTLINE
        # -------------------------

        pen = QPen(Qt.white)
        pen.setWidth(3)
        painter.setPen(pen)

        outline = self.data.get("outline", [])

        if len(outline) > 1:

            for i in range(len(outline)):

                p1 = outline[i]
                p2 = outline[(i + 1) % len(outline)]

                painter.drawLine(
                    QPointF(p1[0], p1[1]),
                    QPointF(p2[0], p2[1])
                )

        # -------------------------
        # CRACKS
        # -------------------------

        pen = QPen(Qt.red)
        pen.setWidth(2)
        painter.setPen(pen)

        for crack in self.data.get("cracks", []):

            for i in range(len(crack) - 1):

                p1 = crack[i]
                p2 = crack[i + 1]

                painter.drawLine(
                    QPointF(p1[0], p1[1]),
                    QPointF(p2[0], p2[1])
                )

        # -------------------------
        # KNOTS
        # -------------------------

        pen = QPen(Qt.yellow)
        brush = QBrush(Qt.yellow)

        painter.setPen(pen)
        painter.setBrush(brush)

        for knot in self.data.get("knots", []):

            x = knot["x"]
            y = knot["y"]
            r = knot.get("radius", 10)

            painter.drawEllipse(
                QPointF(x, y),
                r,
                r
            )

        # -------------------------
        # CUTS
        # -------------------------

        pen = QPen(Qt.green)
        pen.setWidth(4)
        painter.setPen(pen)

        for cut in self.data.get("cuts", []):

            p1 = cut[0]
            p2 = cut[1]

            painter.drawLine(
                QPointF(p1[0], p1[1]),
                QPointF(p2[0], p2[1])
            )

        painter.end()


# -------------------------------------------------
# WebSocket Server
# -------------------------------------------------

viewer = None


async def websocket_handler(websocket):

    print("Client connected")

    try:

        async for message in websocket:

            print("Received:", message)

            data = json.loads(message)

            # Send data to Qt GUI
            viewer.update_data(data)

    except websockets.exceptions.ConnectionClosed:
        print("Client disconnected")


async def websocket_server():

    async with websockets.serve(
        websocket_handler,
        HOST,
        PORT
    ):

        print(f"WebSocket server running at ws://{HOST}:{PORT}")

        await asyncio.Future()


def start_websocket_server():

    asyncio.run(websocket_server())


# -------------------------------------------------
# Main
# -------------------------------------------------

app = QApplication(sys.argv)

viewer = Viewer()
viewer.show()

# Start WebSocket server in another thread
thread = threading.Thread(
    target=start_websocket_server,
    daemon=True
)

thread.start()

sys.exit(app.exec())