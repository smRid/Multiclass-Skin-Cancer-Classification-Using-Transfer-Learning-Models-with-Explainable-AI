import lime
import lime.lime_image
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
import numpy as np
from keras.api.models import load_model
from keras.api.applications.xception import preprocess_input
from keras.api.preprocessing import image


def calculate_iou(pred_mask, gt_mask):
    """
    Calculate Intersection over Union (IoU) between predicted and ground truth masks.

    Args:
        pred_mask (numpy.ndarray): Predicted binary mask (0 or 1 values).
        gt_mask (numpy.ndarray): Ground truth binary mask (0 or 1 values).

    Returns:
        float: IoU score.
    """
    pred_mask = (pred_mask > 0.5).astype(np.uint8)
    gt_mask = (gt_mask > 0.5).astype(np.uint8)
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    return intersection / union if union != 0 else 0.0


def lime_image_example():
    # Load pre-trained model
    model = load_model("inceptionv3_model.keras")
    img_file = "ISIC_0031518.jpg"

    # Load and preprocess the image
    img_path = f"test_image/{img_file}"  # Replace with your image file path
    img = image.load_img(img_path, target_size=(299, 299))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Predict the class
    preds = model.predict(img_array)
    print("Predicted:", preds)

    # Initialize LIME explainer
    explainer = lime.lime_image.LimeImageExplainer()

    # Explain the prediction
    explanation = explainer.explain_instance(img_array[0], model.predict, top_labels=1, hide_color=0, num_samples=1000)

    # Get image and mask for the explanation
    temp, predicted_mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=False)

    # Load ground truth mask
    gt_mask_path = f"Mask_Image/{img_file}"
    gt_mask = image.load_img(gt_mask_path, target_size=(299, 299), color_mode="grayscale")
    gt_mask = image.img_to_array(gt_mask)
    gt_mask = (gt_mask / 255.0).squeeze()

    # Calculate IoU
    iou_score = calculate_iou(predicted_mask, gt_mask)
    print(f"IoU Score: {iou_score:.2f}")

    # Display original image and LIME explanation side by side
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(img)
    axes[0].axis("off")
    axes[0].set_title("Original Image")

    axes[1].imshow(mark_boundaries(temp / 255.0, predicted_mask))
    axes[1].axis("off")
    axes[1].set_title(f"LIME Explanation (IoU: {iou_score:.2f})")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    lime_image_example()
