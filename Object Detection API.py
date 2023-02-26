#kidus Berhanu
import tensorflow as tf
from object_detection.utils import dataset_util
from object_detection.utils import config_util
from object_detection import model_hparams


from object_detection.utils import visualization_utils as vis_util

def load_model(model_path):
    """Loads a pre-trained object detection model from a saved model directory"""
    return tf.saved_model.load(model_path)

def prepare_input_image(image_path):
    """Loads and preprocesses an image for object detection"""
    img = tf.keras.preprocessing.image.load_img(image_path)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    input_img = tf.expand_dims(img_array, axis=0)
    return input_img

def perform_detection(model, input_img):
    """Performs object detection on an input image"""
    output = model.predict(input_img)
    return output

def extract_detection_results(output):
    """Extracts bounding boxes, class labels, and scores from the model output"""
    boxes = output['detection_boxes']
    classes = output['detection_classes']
    scores = output['detection_scores']
    return boxes, classes, scores

def visualize_detection(img_array, boxes, classes, scores, category_index, instance_masks=None, min_score_thresh=0.5, agnostic_mode=False):
    """Visualizes the object detection results on the input image"""
    vis_util.visualize_boxes_and_labels_on_image_array(
        img_array,
        boxes,
        classes,
        scores,
        category_index,
        instance_masks=instance_masks,
        use_normalized_coordinates=True,
        min_score_thresh=min_score_thresh,
        agnostic_mode=agnostic_mode)

def main():
    # load the label map 
    with open('path/to/label_map.pbtxt', 'r') as f:
        label_map = tf.io.gfile.GFile(f.name).read()
    category_index = label_map_util.create_category_index_from_labelmap(label_map, use_display_name=True)

    # Load the pre-trained model
    model_path = "path/to/model"
    model = load_model(model_path)

    # Prepare the input image
    image_path = "path/to/image.jpg"
    input_img = prepare_input_image(image_path)

    # Perform object detection
    output = perform_detection(model, input_img)

    # Extract bounding boxes and class labels from the output
    boxes, classes, scores = extract_detection_results(output)

    # Visualize the detection results
    visualize_detection(img_array, boxes, classes, scores, category_index)

if __name__ == '__main__':
    main()
