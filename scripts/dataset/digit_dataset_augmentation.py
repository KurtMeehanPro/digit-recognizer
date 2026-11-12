"""Training datasets with optional digit rotations."""
# Standard imports
import math

# Third-party imports
import torch


def rotate_digit(image, degrees):
    """Rotate about the image center; square pixels, bilinear, black fill."""
    if degrees == 0:
        return image
    radians = math.radians(degrees)
    c, sine = math.cos(radians), math.sin(radians)
    theta = image.new_tensor([[c, -sine, 0], [sine, c, 0]])

    # Add a leading dimension: (2, 3) becomes (1, 2, 3) for one image transformation.
    theta = theta.unsqueeze(0)

    # For each output pixel, use theta to find the location to sample in the original image.
    grid = torch.nn.functional.affine_grid(theta, (1, *image.shape), align_corners=False)

    return torch.nn.functional.grid_sample(image.unsqueeze(0), grid, mode="bilinear",
                                          padding_mode="zeros", align_corners=False)[0]


class FixedRotationDataset(torch.utils.data.Dataset):
    """Lazy fixed views of training originals; no additional image storage.
    With ten angles you’re getting ten views per image,
    but not storing ten expanded datasets—they’re produced when requested.

    (index // number_of_views) identifies the original
    (index % number_of_views) selects its angle.

    Zero preserves the original tensor.
    Every angle occurs exactly once per original in each full shuffled epoch.
    Split before wrapping; validation must never use this dataset.
    """
    def __init__(self, dataset, angles):
        self.dataset = dataset
        self.angles = tuple(float(angle) for angle in angles)
        if not self.angles or len(set(self.angles)) != len(self.angles) or any(
                not math.isfinite(angle) or abs(angle) > 180 for angle in self.angles):
            raise ValueError("Fixed rotation angles must be unique finite degrees within [-180, 180]")

    def __len__(self):
        return len(self.dataset) * len(self.angles)

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        original, view = divmod(index, len(self.angles))
        image, label = self.dataset[original]
        return rotate_digit(image, self.angles[view]), label


class RandomRotationDataset(torch.utils.data.Dataset):
    """Training-only CPU rotations with a separate reproducible random stream.

    Uniform angles in [-degrees, degrees]; bilinear sampling, black fill (0.0),
    align_corners=False, unchanged 28x28 frame. Small rotations intend to preserve
    labels but can crop strokes; validation is never wrapped. num_workers=0 keeps
    this stream deterministic across the complete run, without epoch resets.
    """
    def __init__(self, dataset, degrees, seed):
        if not math.isfinite(degrees) or not 0 <= degrees <= 180:
            raise ValueError("Rotation degrees must be finite and between 0 and 180")
        self.dataset = dataset
        self.degrees = float(degrees)
        self.generator = torch.Generator().manual_seed(seed)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, label = self.dataset[index]

        # we generate a random angle between [-degrees, degrees] for each image
        angle = (torch.rand((), generator=self.generator).item() * 2 - 1) * self.degrees

        # we ask for a rotated image, but we don't change the label
        return rotate_digit(image, angle), label


def augment_dataset(dataset, augmentation_seed=42, rotation_angles=None,
                     rotation_degrees=None):
    """Take a training dataset, optionally rotate it, and return a dataset.

    dataset: The original training images and labels. Pass only training data;
    validation stays unchanged.

    augmentation_seed: The shared augmentation seed for reproducible random
    choices. Fixed views use their exact angles and need no random choices.

    rotation_angles: The exact angles to use as fixed views of each image.
    Supply a nonempty list of unique finite degrees within [-180, 180], or None.
    Fixed views cannot be combined with nonzero rotation_degrees.

    rotation_degrees: The maximum random rotation in each direction, or None.
    For example, 10 lets the code randomly choose an angle between 10 degrees
    left and 10 degrees right. A value of 160 allows -160 to +160 degrees, a
    total range of 320 degrees; 180 allows a full circle, 360 degrees.

    With both rotation_angles and rotation_degrees set to None, return the
    original dataset unchanged. Explicit zero degrees also leaves it unchanged
    when no fixed angles are supplied."""

    if rotation_degrees is None and rotation_angles is None:
        return dataset

    if rotation_angles is None and rotation_degrees == 0:
        return dataset

    if rotation_angles is not None and rotation_degrees:
        raise ValueError("Choose fixed angles or random rotations, not both")

    if rotation_angles is None and (
            not math.isfinite(rotation_degrees) or not 0 <= rotation_degrees <= 180):
        raise ValueError("Rotation degrees must be finite and between 0 and 180")

    # Use the specific rotation angles provided.
    if rotation_angles is not None:
        return FixedRotationDataset(dataset, rotation_angles)

    # Let the code randomly choose rotation angles within the left-and-right limits.
    return RandomRotationDataset(dataset, rotation_degrees, augmentation_seed)


def rotation_configuration(degrees=0.0, seed=42, angles=None):
    """Describe the rotation settings in a dictionary for saving with a run.

    degrees: The maximum random rotation in each direction. For example, 10
    means angles between -10 and +10 degrees. Zero records no rotation when
    angles is None. This value is ignored when fixed angles are supplied.

    seed: The seed recorded for reproducible random angle choices. This
    function only records it; fixed angles do not need random choices.

    angles: The exact fixed angles in degrees to record for each original
    image, or None to use degrees. The returned dictionary stores a list of
    these angles and the number of views per image.

    This function records settings; it does not rotate images or validate the
    values. The dataset wrappers perform validation when they are created.
    """

    # Record the exact fixed views and how they are produced when requested.
    if angles is not None:
        return {"type": "fixed_rotation_views", "angles_degrees": list(angles),
                "views_per_original": len(angles), "generation": "lazy_original_major",
                "interpolation": "bilinear", "fill": 0.0, "align_corners": False,
                "rotation_center": "image_center", "seed": seed, "training_only": True,
                "num_workers": 0, "split_before_augmentation": True}

    # Record the random rotation range, or no rotation when degrees is zero.
    return {"type": "random_rotation" if degrees else "none", "degrees": float(degrees),
            "angle_distribution": "uniform_symmetric", "interpolation": "bilinear",
            "fill": 0.0, "align_corners": False, "seed": seed, "training_only": True,
            "num_workers": 0}
