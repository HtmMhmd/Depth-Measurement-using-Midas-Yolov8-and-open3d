import numpy as np

def calculate_distance(bbox):
    """
    Calculates the distance based on the given bounding box.

    Parameters:
    bbox (tuple): A tuple containing the bounding box coordinates.

    Returns:
    float: The calculated distance.
    """

    y = [90, 30, 20] # collected data by moving the box to multiple different distances
    x = [107, 231, 455]

    coefficients = np.polyfit(x, y, 3)
    poly_function = np.poly1d(coefficients)
    dist = poly_function((bbox[1] - bbox[0])[0])

    return dist