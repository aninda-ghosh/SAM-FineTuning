import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

def build_point_grid(n_per_side: int) -> np.ndarray:
    """Generates a 2D grid of points evenly spaced in [0,1]x[0,1]."""
    offset = 1 / (2 * n_per_side)
    points_one_side = np.linspace(offset, 1 - offset, n_per_side)
    points_x = np.tile(points_one_side[None, :], (n_per_side, 1))
    points_y = np.tile(points_one_side[:, None], (1, n_per_side))
    points = np.stack([points_x, points_y], axis=-1).reshape(-1, 2)
    return points

def get_bbox_point_and_inputlabel_prompts(pixel_masks, image_width, image_height, num_boxes, min_distance, box_percentage):
    bbox_prompts = []
    point_prompts = []
    input_labels_prompts = []
    
    if len(pixel_masks) == 0:
        # TODO: We have to create few bounding boxes here with just blank background
        bbox_prompts = generate_random_bounding_boxes(image_width, image_height, num_boxes, min_distance, box_percentage)
    else:
        for mask in pixel_masks:
            # get bounding box from mask
            y_indices, x_indices = np.where(mask > 0)

            # # This is done if we have masks but the labels are all 0, we have to generate random bounding boxes then
            # if (len (x_indices) == 0) or (len(y_indices) == 0):
            #     bbox_prompts = generate_random_bounding_boxes(image_width, image_height, num_boxes, min_distance, box_percentage)
            # else:
            x_min, x_max = np.min(x_indices), np.max(x_indices)
            y_min, y_max = np.min(y_indices), np.max(y_indices)
            # add perturbation to bounding box coordinates
            H, W = mask.shape
            x_min = max(0, x_min - np.random.randint(0, 30))
            x_max = min(W, x_max + np.random.randint(0, 30))
            y_min = max(0, y_min - np.random.randint(0, 30))
            y_max = min(H, y_max + np.random.randint(0, 30))
            bbox = [x_min, y_min, x_max, y_max]
            bbox_prompts.append(bbox)

            # Get grid points within the bounding box
            point_grid_per_crop= build_point_grid(10)
            #This forms a 2D grid points and we need to normalize it to the bbox size
            point_grid_per_crop[:, 0] = point_grid_per_crop[:, 0] * (x_max - x_min) + x_min
            point_grid_per_crop[:, 1] = point_grid_per_crop[:, 1] * (y_max - y_min) + y_min
            #Convert the grid points to a integer
            point_grid_per_crop = np.around(point_grid_per_crop).astype(np.float64)
            points_per_crop = np.array([np.array([point_grid_per_crop])])
            points_per_crop = np.transpose(points_per_crop, axes=(0, 2, 1, 3))
            point_prompts.append(points_per_crop)

            input_labels = np.ones_like(points_per_crop[:, :, :, 0], dtype=np.int64)
            input_labels_prompts.append(input_labels)

    return bbox_prompts, point_prompts, input_labels_prompts



# To Generate the Random Points and Bounding Boxes using https://www.jasondavies.com/poisson-disc/ algorithm
def generate_random_bounding_boxes(image_width, image_height, number_boxes, min_distance, box_percentage):
    def generate_random_points(image_width, image_height, min_distance, num_points, k=30):
        image_size = (image_width, image_height)
        cell_size = min_distance / np.sqrt(2)
        grid_width = int(np.ceil(image_width / cell_size))
        grid_height = int(np.ceil(image_height / cell_size))
        grid = np.empty((grid_width, grid_height), dtype=np.int32)
        grid.fill(-1)

        points = []
        active_points = []

        def generate_random_point():
            return np.random.uniform(0, image_width), np.random.uniform(0, image_height)

        def get_neighboring_cells(point):
            x, y = point
            x_index = int(x / cell_size)
            y_index = int(y / cell_size)

            cells = []
            for i in range(max(0, x_index - 2), min(grid_width, x_index + 3)):
                for j in range(max(0, y_index - 2), min(grid_height, y_index + 3)):
                    cells.append((i, j))

            return cells

        def is_point_valid(point):
            x, y = point
            if x < 0 or y < 0 or x >= image_width or y >= image_height:
                return False

            x_index = int(x / cell_size)
            y_index = int(y / cell_size)

            cells = get_neighboring_cells(point)
            for cell in cells:
                if grid[cell] != -1:
                    cell_points = points[grid[cell]]
                    if np.any(np.linalg.norm(np.array(cell_points) - np.array(point), axis=None) < min_distance):
                        return False

            return True

        def add_point(point):
            x, y = point
            x_index = int(x / cell_size)
            y_index = int(y / cell_size)

            points.append(point)
            index = len(points) - 1
            grid[x_index, y_index] = index
            active_points.append(point)

        start_point = generate_random_point()
        add_point(start_point)

        while active_points and len(points) < num_points:
            random_index = np.random.randint(len(active_points))
            random_point = active_points[random_index]
            added_new_point = False

            for _ in range(k):
                angle = 2 * np.pi * np.random.random()
                radius = min_distance + min_distance * np.random.random()
                new_point = (random_point[0] + radius * np.cos(angle), random_point[1] + radius * np.sin(angle))
                if is_point_valid(new_point):
                    add_point(new_point)
                    added_new_point = True

            if not added_new_point:
                active_points.pop(random_index)

        return points
    

    points = generate_random_points(image_width, image_height, min_distance, number_boxes)
    
    
    box_width = int(image_width * box_percentage)
    box_height = int(image_height * box_percentage)

    bounding_boxes = []
    for point in points:
        x = int(point[0] - box_width / 2)
        y = int(point[1] - box_height / 2)

        # Adjust the coordinates to keep the bounding box within the image
        x = max(0, min(x, image_width - box_width))
        y = max(0, min(y, image_height - box_height))

        bounding_boxes.append([x, y, x+box_width, y+box_height])

    return bounding_boxes

# def visualize_points_and_boxes(image_width, image_height, bounding_boxes):
#     # Create a figure and axis
#     fig, ax = plt.subplots(1)
#     ax.set_xlim(0, image_width)
#     ax.set_ylim(0, image_height)

#     # # Plot the points
#     # for point in points:
#     #     ax.plot(point[0], point[1], 'ro')

#     # Plot the bounding boxes
#     for bbox in bounding_boxes:
#         rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1], linewidth=1, edgecolor='g', facecolor='none')
#         ax.add_patch(rect)

#     # Display the image
#     plt.show()