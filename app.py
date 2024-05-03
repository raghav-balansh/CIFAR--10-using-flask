from flask import Flask, render_template, request
import tensorflow as tf
from werkzeug.utils import secure_filename
import numpy as np
from PIL import Image


app = Flask(__name__)

# uploading the model
model = tf.keras.models.load_model('cifar_10.h5')


# main branch for flask app
@app.route('/')
def index():
    return render_template('index.html')


# Prediction function in this
@app.route('/', methods=['POST'])
def submit_image():
    uploaded_img = request.files['image']   # Requesting for the file in this

    if uploaded_img.filename != '':
        # we have assign the filename as global because in my system there an error occur
        global filename
        filename = secure_filename(uploaded_img.filename)
        print(filename)
        # save the filein Static Folder
        uploaded_img.save('static/' + filename)

        # Introducing the try-except block for error Handling
        try:
            img = Image.open('static/' + filename)
            img = tf.keras.preprocessing.image.smart_resize(img, (32, 32))   # Resizing the image as per google

            # change this image into array format
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            img_array = img_array / 255.0
            img_array = img_array.reshape(1, 32, 32, 3)     # create for 1 kernel and 32x32 image and 3 channels r,g,b

            # giving the image of array to model to prediction
            prediction = model.predict(img_array)
            # get the class of images
            predicted_class = np.argmax(prediction[0])

            class_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
            predicted_label = class_labels[predicted_class]

        # handling input output error and os based error
        except (IOError, OSError) as e:
            error_message = f"Error processing image: {str(e)}"
            return render_template('index.html', prediction=error_message)

        # handling value error and Arithmetic Errors
        except (ValueError, ArithmeticError):
            return render_template('index.html', prediction='Invalid Image or can not reshape the images')

        # sending the response to the image prediction
        return render_template('index.html', prediction=predicted_label, filename=filename)

    # set to the file to none and ready for second creation
    return render_template('index.html', prediction="No file uploaded")


# main for running
if __name__ == '__main__':
    app.run(debug=True)
