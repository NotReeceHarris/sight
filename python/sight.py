from imageai.Detection import ObjectDetection 

recognizer = ObjectDetection()
path_model = "./models/yolo-tiny.h5"
path_input = "./input/image.jpg"
path_output = "./output/newimage.jpg"

recognizer.setModelTypeAsTinyYOLOv3()
recognizer.setModelPath(path_model)
recognizer.loadModel()

recognition = recognizer.detectObjectsFromImage(  
    input_image = path_input,  
    output_image_path = path_output  
)

for eachItem in recognition:  
    print(eachItem["name"] , " : ", eachItem["percentage_probability"])  