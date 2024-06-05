import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--mode', default='image', help='Input Mode - image/video')
args = parser.parse_args()

def main():
    import warnings
    warnings.filterwarnings('ignore')

    import numpy as np
    import os
    import six.moves.urllib as urllib
    import sys
    import tarfile
    import tensorflow as tf
    import zipfile
    import cv2
    import glob

    from distutils.version import StrictVersion
    from collections import defaultdict
    from io import StringIO
    from matplotlib import pyplot as plt
    from PIL import Image

    from object_detection.utils import label_map_util
    from object_detection.utils import visualization_utils as vis_util

    # This is needed since the notebook is stored in the object_detection folder.
    ##sys.path.append("..")
    from object_detection.utils import ops as utils_ops

    print(f'TF Version: {(tf.__version__)}')
    if StrictVersion(tf.__version__) < StrictVersion('1.12.0'):
        raise ImportError('Please upgrade your TensorFlow installation to v1.12.*.')

    print(tf.__version__)

    MODEL_NAME = 'wood_inference_graph'
    MODEL_FILE = f'{MODEL_NAME}.tar.gz'
    # DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

    # Path to frozen detection graph. This is the actual model that is used for the object detection.
    PATH_TO_FROZEN_GRAPH = f'{MODEL_NAME}/frozen_inference_graph.pb'

    # List of the strings that is used to add correct label for each box.
    PATH_TO_LABELS = os.path.join('data', 'wood_label_map.pbtxt')
    
    # opener = urllib.request.URLopener()
    # opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
    # tar_file = tarfile.open(MODEL_FILE)
    # for file in tar_file.getmembers():
    #     file_name = os.path.basename(file.name)
    #     if 'frozen_inference_graph.pb' in file_name:
    #         tar_file.extract(file, os.getcwd())

    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

    def load_image_into_numpy_array(image):
        (im_width, im_height) = image.size
        return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

    # PATH_TO_TEST_IMAGES_DIR = 'test_images'
##    TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(1, 7) ]
    # TEST_IMAGE_PATHS = glob.glob(f'{PATH_TO_TEST_IMAGES_DIR}/*')

    # Size, in inches, of the output images.
    # IMAGE_SIZE = (12, 8)

    def run_inference_for_single_image(image, graph):
        with graph.as_default():
            with tf.Session() as sess:
                # Get handles to input and output tensors
                ops = tf.get_default_graph().get_operations()
                all_tensor_names = {output.name for op in ops for output in op.outputs}
                tensor_dict = {}
                for key in ['num_detections', 'detection_boxes', 'detection_scores',
                          'detection_classes', 'detection_masks']:
                    tensor_name = key + ':0'
                    if tensor_name in all_tensor_names:
                        tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)
                if 'detection_masks' in tensor_dict:
                    # The following processing is only for single image
                    detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                    detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
                    # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                    real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
                    detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                    detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                    detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                        detection_masks, detection_boxes, image.shape[0], image.shape[1])
                    detection_masks_reframed = tf.cast(
                        tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                    # Follow the convention by adding back the batch dimension
                    tensor_dict['detection_masks'] = tf.expand_dims(
                        detection_masks_reframed, 0)
                image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

                # Run inference
                output_dict = sess.run(tensor_dict,
                                 feed_dict={image_tensor: np.expand_dims(image, 0)})

                # all outputs are float32 numpy arrays, so convert types as appropriate
                output_dict['num_detections'] = int(output_dict['num_detections'][0])
                output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.uint8)
                output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
                output_dict['detection_scores'] = output_dict['detection_scores'][0]
                if 'detection_masks' in output_dict:
                    output_dict['detection_masks'] = output_dict['detection_masks'][0]
        return output_dict

    def video_detection():
        # Object Detection on Video Stream from Webcam
        cap = cv2.VideoCapture(0)

        # Define the codec and create VideoWriter object
    ##    fourcc = cv2.VideoWriter_fourcc(*'FMP4')
    ##    out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (640, 480))

        while True:
            ret, frame = cap.read()
            
            if ret==True:
##                print("frame.shape", "="*70, frame.shape)
##                image = Image.fromarray(frame)
##                print("fromarray", "="*70, image.size)
##                image_np = load_frame_into_numpy_array(image)
##                print("load_frame_into_numpy_array", "="*70, image_np.size)
##                image_np_expanded = np.expand_dims(frame, axis=0)
##                print("expand_dims", "="*70, image_np_expanded.shape)

                output_dict = run_inference_for_single_image(frame, detection_graph)
                vis_util.visualize_boxes_and_labels_on_image_array(
                frame,
                output_dict['detection_boxes'],
                output_dict['detection_classes'],
                output_dict['detection_scores'],
                category_index,
                instance_masks=output_dict.get('detection_masks'),
                use_normalized_coordinates=True,
                line_thickness=8)
                
                cv2.imshow("Object Detection", frame)
    ##            out.write(image_np)
                
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break
            else:
                break

        cap.release()
##        out.release()
        cv2.destroyAllWindows()

##    video_detection()

    def video_detection_2():
        
        cap = cv2.VideoCapture(0)
        # Detection
        with detection_graph.as_default():
            with tf.Session(graph=detection_graph) as sess:
                while True:
                    # Read frame from camera
                    ret, image_np = cap.read()

                    if ret == True:
                        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                        image_np_expanded = np.expand_dims(image_np, axis=0)
                        # Extract image tensor
                        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                        # Extract detection boxes
                        boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                        # Extract detection scores
                        scores = detection_graph.get_tensor_by_name('detection_scores:0')
                        # Extract detection classes
                        classes = detection_graph.get_tensor_by_name('detection_classes:0')
                        # Extract number of detectionsd
                        num_detections = detection_graph.get_tensor_by_name(
                            'num_detections:0')
                        # Actual detection.
                        (boxes, scores, classes, num_detections) = sess.run(
                            [boxes, scores, classes, num_detections],
                            feed_dict={image_tensor: image_np_expanded})
                        # Visualization of the results of a detection.
                        vis_util.visualize_boxes_and_labels_on_image_array(
                            image_np,
                            np.squeeze(boxes),
                            np.squeeze(classes).astype(np.int32),
                            np.squeeze(scores),
                            category_index,
                            use_normalized_coordinates=True,
                            line_thickness=8)

                        # Display output
                        cv2.imshow('object detection', cv2.resize(image_np, (800, 600)))

                        if cv2.waitKey(25) & 0xFF == ord('q'):
        ##                    cv2.destroyAllWindows()
                            break
                    else:
                        break

        cap.release()
##        out.release()
        cv2.destroyAllWindows()

    def image_detection():
        PATH_TO_TEST_IMAGES_DIR = 'test_images'
        ##    TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(1, 7) ]
        TEST_IMAGE_PATHS = glob.glob(f'{PATH_TO_TEST_IMAGES_DIR}/*')
    
        # Size, in inches, of the output images.
        IMAGE_SIZE = (12, 8)
        
        # Results from faster_rcnn_nas pre-built model
        for image_path in TEST_IMAGE_PATHS:
            print(f'Processing {os.path.basename(image_path)}...')
            image = Image.open(image_path)
            image = image.resize((480, 600), Image.ANTIALIAS)
            
            # the array based representation of the image will be used later in order to prepare the
            # result image with boxes and labels on it.
##            print("image.shape", "="*70, image.size)
            image_np = load_image_into_numpy_array(image)
##            print("load_frame_into_numpy_array", "="*70, image_np.shape)
            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
##            image_np_expanded = np.expand_dims(image_np, axis=0)
##            print("expand_dims", "="*70, image_np_expanded.shape)

            # Actual detection.
            output_dict = run_inference_for_single_image(image_np, detection_graph)
            # Visualization of the results of a detection.
            vis_util.visualize_boxes_and_labels_on_image_array(
              image_np,
              output_dict['detection_boxes'],
              output_dict['detection_classes'],
              output_dict['detection_scores'],
              category_index,
              instance_masks=output_dict.get('detection_masks'),
              use_normalized_coordinates=True,
              line_thickness=8)
            im = Image.fromarray(image_np)
            im.save(f'detections/{os.path.basename(image_path)}')
            plt.figure(figsize=IMAGE_SIZE)
            plt.imshow(image_np)

    # image_detection()
    if args.mode == 'image':
        image_detection()
    else:
        video_detection_2()

if __name__ == '__main__':
    main()
