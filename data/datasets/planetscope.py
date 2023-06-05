# encoding: utf-8
"""
@author:  Aninda Ghosh
@contact: aghosh57@asu.edu
"""

import numpy as np
import cv2
import geopandas as gpd
from torch.utils.data import Dataset

class ParcelDataset(Dataset):
    """Parcel dataset
    --------------------------------
    This class is used to load the dataset.
    The data is stored in a folder with the following structure:
    - data
        - parcels
            - xxxxx.png
            - xxxxx.geojson
            - ...

    The labels is loaded using the `geopandas` library.
    The images are loaded using the `matplotlib.pyplot` module.
    The labels are stored in boolean masks. The masks are stored in a list.
    """
    
    def __init__(self, path):
        """
        Args:
            path (string): Path to the folder containing the dataset.
            The dataset is stored in a single folder.
        """
        self.path = path
        # Read the merged parcel data geo json file, This contains the parcel id and the geometry of all the parcels
        self.data = gpd.read_file(self.path + "parcel_data.geojson")
        # Get the image paths, pixel masks and labels
        self.data = self._get_image_prompts_labels(self.data)
        
        self.length = len(self.data)
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        """
        This function returns one sample from the dataset.

        Args: idx (list): List of indexes.

        Returns:
            np.array: Image as data
            np.array: List of box prompts.
            np.array: List of point prompts.
            np.array: List of pixel masks.
        """
        # TODO: Need to fix this function to return single image, bbox prompt, point prompt and pixel mask            
        image_path = self.data[idx][0]
        # Read the image    
        image = cv2.imread(image_path)
        image = np.array(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        bbox_prompts = self.data[idx][1]
        point_prompts = self.data[idx][2]
        labels = self.data[idx][3]

        data = {
            "image": image,
            "bbox_prompts": bbox_prompts,
            "point_prompts": point_prompts,
            "labels": labels,
        }

        return data

    def _convert_polygons_to_pixels(self, parcel_geometry, polygon_labels, size):
        # Create a black background
        background = np.zeros(size, dtype=np.uint8)
         
        # Get the polygon bounds
        min_lon, min_lat, max_lon, max_lat = parcel_geometry.bounds
        
        # Get the width and height of the parcel
        height = max_lat - min_lat
        width = max_lon - min_lon

        # Get the pixel width and height
        pixel_height = size[0]/height
        pixel_width = size[1]/width

        pixel_masks = []
        for label in polygon_labels.geometry:
            pixel_mask = background.copy()
            # Iterate over the polygons in the multipolygon
            for polygon in label.geoms:
                coords = polygon.exterior.coords.xy

                # Get the pixel coords
                pixel_coords = []
                for i in range(len(coords[0])):
                    x = coords[0][i]
                    y = coords[1][i]

                    # Get the pixel x and y
                    pixel_x = (x - min_lon) * pixel_width
                    pixel_y = (max_lat - y) * pixel_height

                    pixel_coords.append([pixel_x, pixel_y])

                # Convert to int to make it work with cv2
                pixel_coords = np.array(pixel_coords, dtype=np.int32)
                # Fill the polygons and append to the pixel mask
                cv2.fillPoly(pixel_mask, pts=[pixel_coords], color=255)
            pixel_masks.append(pixel_mask.astype(dtype=bool))
        
        # Return the pixel masks as a numpy array
        return np.array(pixel_masks)    

    def _get_bbox_and_point_prompts(self, pixel_masks):
        bbox_prompts = []
        point_prompts = []
        for mask in pixel_masks:
            # get bounding box from mask
            y_indices, x_indices = np.where(mask > 0)
            x_min, x_max = np.min(x_indices), np.max(x_indices)
            y_min, y_max = np.min(y_indices), np.max(y_indices)
            # add perturbation to bounding box coordinates
            H, W = mask.shape
            x_min = max(0, x_min - np.random.randint(0, 20))
            x_max = min(W, x_max + np.random.randint(0, 20))
            y_min = max(0, y_min - np.random.randint(0, 20))
            y_max = min(H, y_max + np.random.randint(0, 20))
            bbox = [x_min, y_min, x_max, y_max]
            bbox_prompts.append(bbox)

            # Get grid points within the bounding box
            x = np.linspace(x_min, x_max, int((x_max-x_min)/10))
            y = np.linspace(y_min, y_max, int((y_max-y_min)/10))

            point_prompts.append(list((int(_x),int(_y)) for _x,_y in zip(x,y)))

        return bbox_prompts, point_prompts

    def _get_image_prompts_labels(self, parcel_data, size=(448, 448)):
        """Convert the polygon masks to pixel masks
        Note:
            Refrain from sending image sizes more than 1000x1000. This can cause some issues with SAM image encoder.
        Args:
            parcel_data (geopandas.GeoDataFrame): GeoDataFrame containing the parcel data.

        Returns:
            image_list: List of image paths.
            list: List of pixel masks.
        """
        
        # Store the processed data in a list
        processed_data = []
        
        for _, data in parcel_data.iterrows():
            parcel_id = data['parcel_id']
            geometry = data['geometry']
            
            polygon_mask_list = gpd.read_file(self.path + parcel_id + '.geojson')
            
            # TODO: This is an ambiguous step, need to figure out how to handle this, Ask Hannah about this
            # If the parcel has no labels, let's keep it with no labels and no prompts
            if len(polygon_mask_list) == 0:
                # Store the image path based on the current geoJSON file name only if the parcel has labels
                processed_data.append((self.path + parcel_id + '.png', [], [], []))
                continue
            
            # Get the pixel masks for the parcel
            pixel_masks = self._convert_polygons_to_pixels(geometry, polygon_mask_list, size)
            # Get the bounding boxes and point prompts for the parcel
            bounding_boxes, point_prompts = self._get_bbox_and_point_prompts(pixel_masks)

            processed_data.append((self.path + parcel_id + '.png', bounding_boxes, point_prompts, pixel_masks))

        return processed_data