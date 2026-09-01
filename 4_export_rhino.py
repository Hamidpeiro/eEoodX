"""
Step 4: Turn a measurement JSON (from measure_board.py) into a Rhino .3dm
file - a closed polyline for the board outline, plus a text label with its
dimensions and color reading. Coordinates are the same real-world mm frame
your homography defined, so multiple boards will lay out correctly relative
to each other if you accumulate them on one layer/file.

This does NOT require Rhino to be installed - rhino3dm is a standalone
library that reads/writes .3dm files directly.

Usage:
    python 4_export_rhino.py sample_output/board1_measurement.json
    python 4_export_rhino.py sample_output/board1_measurement.json sample_output/board2_measurement.json  # combine into one file
"""

import rhino3dm
import json
import sys
import os
import config


def add_board_to_model(model, data, z_offset=0.0):
    corners = data["corners_mm"]
    pts = [rhino3dm.Point3d(x, y, z_offset) for x, y in corners]
    pts.append(pts[0])  # close the loop

    polyline = rhino3dm.Polyline(pts)
    curve = polyline.ToPolylineCurve()
    model.Objects.AddCurve(curve)

    # Label with dimensions + color, placed near the board's first corner
    label_text = (
        f"{data['length_mm']:.1f} x {data['width_mm']:.1f} mm  "
        f"LAB({data['color_lab']['L']:.0f},{data['color_lab']['a']:.0f},{data['color_lab']['b']:.0f})"
    )
    model.Objects.AddTextDot(label_text, pts[0])

    # Defect markers, if the JSON has any (mm coordinates, same frame)
    for defect in data.get("defects", []):
        dpt = rhino3dm.Point3d(defect["x_mm"], defect["y_mm"], z_offset)
        model.Objects.AddPoint(dpt)


def main():
    if len(sys.argv) < 2:
        print("Usage: python 4_export_rhino.py measurement1.json [measurement2.json ...]")
        raise SystemExit(1)

    model = rhino3dm.File3dm()
    model.Settings.ModelUnitSystem = rhino3dm.UnitSystem.Millimeters

    for i, json_path in enumerate(sys.argv[1:]):
        with open(json_path) as f:
            data = json.load(f)
        # Stagger multiple boards apart on Y so they don't overlap in the
        # viewer if you're exporting several at once - purely visual, delete
        # this if your mm coordinates already place them correctly.
        add_board_to_model(model, data, z_offset=0.0)
        print(f"Added {json_path}: {data['length_mm']}mm x {data['width_mm']}mm")

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(config.OUTPUT_DIR, "boards.3dm")
    model.Write(out_path, 7)
    print(f"\nSaved {out_path}")
    print("Open this directly in Rhino, or drag-drop into Rhino.Inside/Grasshopper.")


if __name__ == "__main__":
    main()
