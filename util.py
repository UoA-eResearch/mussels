#!/usr/bin/env python3

from segment_anything import SamAutomaticMaskGenerator, sam_model_registry, SamPredictor
import cv2  # Image processing
import numpy as np  # Numeric data
import pandas as pd  # Tabular data
import matplotlib.pyplot as plt # Plotting
plt.rcParams['figure.figsize'] = [10, 10]
from rasterio.features import shapes  # Vectorising rasters to polygons
from shapely import Point, LineString, Polygon  # Geometry
from shapely.geometry import box, shape
from shapely.ops import nearest_points
from shapely import centroid
import geopandas as gpd  # Plotting polygons
from tqdm.auto import tqdm  # Progress bars
import time

tqdm.pandas()


def load_SAM():
    checkpoint = "sam_vit_h_4b8939.pth"
    sam = sam_model_registry["vit_h"](checkpoint=checkpoint)
    # SAM takes 1 min 18 s on a CPU and 15.6s on a P40 GPU, so best to use a GPU
    device = "cuda:0"
    sam.to(device)
    return sam


def snap(g1, g2, threshold=1e6):
    coordinates = []
    for x, y in g1.coords:  # for each vertex in the first line
        point = Point(x, y)
        p1, p2 = nearest_points(point, g2)  # find the nearest point on the second line
        if p1.distance(p2) <= threshold:
            # it's within the snapping tolerance, use the snapped vertex
            coordinates.append(p2.coords[0])
        else:
            # it's too far, use the original vertex
            coordinates.append((x, y))
    # convert coordinates back to a LineString and return
    return LineString(coordinates)


def load_img(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)


def rectify(sam, img):
    # Reproject trapezoidal tray to rectangular (rectify)
    masks = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32,  # The number of points to be sampled along one side of the image
        min_mask_region_area=100,  # If >0, postprocessing will be applied to remove disconnected regions and holes in masks with area smaller than min_mask_region_area
    ).generate(img)
    tray = max(masks, key=lambda x: x["area"])
    # Convert binary mask to polygon
    tray = shape(
        next(shapes(tray["segmentation"].astype(np.uint8), mask=tray["segmentation"]))[
            0
        ]
    )
    trapezoid = snap(tray.envelope.exterior, tray)
    bounds = trapezoid.envelope.exterior
    source_corners = np.float32(trapezoid.coords[:4])
    target_corners = np.float32(bounds.coords[:4])
    matrix = cv2.getPerspectiveTransform(source_corners, target_corners)
    img = cv2.warpPerspective(
        img,
        matrix,
        (img.shape[1], img.shape[0]),
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    return img


def get_shapes(sam, img):
    masks = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=50,  # The number of points to be sampled along one side of the image
        pred_iou_thresh=0.88,  # A filtering threshold in [0,1], using the model's predicted mask quality.
        stability_score_thresh=0.95,  # The amount to shift the cutoff when calculated the stability score.
        box_nms_thresh=0.7,  # The box IoU cutoff used by non-maximal suppression to filter duplicate masks.
        crop_nms_thresh=0.7,  # The box IoU cutoff used by non-maximal suppression to filter duplicate masks between different crops.
        crop_n_layers=0,  #  If >0, mask prediction will be run again on crops of the image. Sets the number of layers to run, where each layer has 2**i_layer number of image crops.
        crop_n_points_downscale_factor=1,  # The number of points-per-side sampled in layer n is scaled down by crop_n_points_downscale_factor**n.
        min_mask_region_area=100,  # If >0, postprocessing will be applied to remove disconnected regions and holes in masks with area smaller than min_mask_region_area
    ).generate(img)
    masks = sorted(masks, key=(lambda x: x["area"]), reverse=True)
    full_mask = np.zeros_like(masks[0]["segmentation"]).astype(int)
    for i in range(len(masks)):
        x, y = np.where(masks[i]["segmentation"])
        full_mask[x, y] = i + 1
    # Vectorise raster
    shape_gen = (
        (shape(s), v) for s, v in shapes(full_mask.astype(np.uint8), mask=full_mask > 0)
    )
    # Convert shapes to GeoDataFrame, taking CRS from the image
    df = gpd.GeoDataFrame(dict(zip(["geometry", "id"], zip(*shape_gen))))
    df["area"] = df.area
    df = df.sort_values(by="area", ascending=False)
    return df


def filter_to_tray(df):
    tray = df.iloc[0]
    return df[df.within(tray.geometry.envelope.buffer(-5)) & (df.area > 550)]


def get_px_per_cm(ruler):
    x1, y1, x2, y2 = ruler.geometry.bounds
    ruler_height = y2 - y1
    # pixels to cm conversion. ruler is 32cm long
    px_per_cm = ruler_height / 32
    return px_per_cm


# Get diameter of polygon by brute force, checking each point pair
def get_diameter(poly):
    max_dist = 0
    coords = [Point(x, y) for x, y in poly.exterior.coords]
    result_coords = []
    for i, a in enumerate(coords):
        for j, b in enumerate(coords):
            if i < j:
                dist = a.distance(b)
                if dist > max_dist:
                    max_dist = dist
                    result_coords = [a, b]

    line = LineString(result_coords)
    return line

def annotate_length(row):
    x, y = centroid(row.geometry).coords[0]
    plt.text(s=f"{row.length_cm:.2f}cm", x=x, y=y, ha='center', va='center', bbox=dict(boxstyle="round", alpha=.6), fontsize="x-small")

def measure_mussels_in_image(sam, filepath, plot=False):
    img = load_img(filepath)
    img = rectify(sam, img)
    df = get_shapes(sam, img)
    ruler = df.iloc[2]
    px_per_cm = get_px_per_cm(ruler)
    df = filter_to_tray(df)
    df["diameter_line"] = df.geometry.apply(get_diameter)
    df["length_cm"] = df.diameter_line.length / px_per_cm
    if plot:
        plt.imshow(img)
        ax = plt.gca()
        ruler.diameter_line = get_diameter(ruler.geometry)
        ruler.length_cm = ruler.diameter_line.length / px_per_cm
        gpd.GeoSeries(ruler.diameter_line).plot(color="cyan", ax=ax)
        annotate_length(ruler)
        df.diameter_line.plot(color="cyan", ax=ax)
        for i, row in df.iterrows():
            annotate_length(row)
        plt.tight_layout()
        plt.savefig(filepath.replace(".png", "_measured.png"))
    return df.length_cm.describe()

if __name__ == "__main__":
    start = time.time()
    sam = load_SAM()
    print(f"{round(time.time() - start)}s: SAM loaded")
    print(measure_mussels_in_image(sam, "test.png", plot=True))