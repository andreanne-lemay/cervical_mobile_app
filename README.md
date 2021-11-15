# Cervical Classifier App

```bash
git clone https://github.com/andreanne-lemay/cervical_mobile_app.git
```

## Run the Android App
The Android app `CervicalApp` is based on the demo available [here](https://github.com/pytorch/android-demo-app/tree/master/HelloWorldApp). In order to preserve the confidentiality of the medical data, a dummy model trained on ImageNet and a dummy image with the same dimensions as one of the medical images available were included in the `assets` folder.

The app does the following steps:
1. Load the image from the assets folder (`dummy_img.jpg`)
2. Crop the image according hard-coded bounding box coordinates which will eventually come from the object detection model.
3. Resize the image to 256x256 to match the processing pipeline done during training.
4. Convert the image to a tensor.
5. Load the PyTorch mobile model (`dummy_model.ptl`).
6. Run inference.

To simulate multiple forward passes describing the Monte Carlo inference approach, the following line in `MainActivity.java` can be modifed.

```java
int mcIt = 0; // Can be set to ~20 to verify MC model runtime. When mcIt is set to 0 it will simply do one forward pass.
```

Using Android studio with a phone emulator or a real device with the app installed, run the app located in the `CervicalApp` folder.

The classified image should appear on the screen. The dummy image is a set of randomly assigned pixel from 0 to 255. The prediction (Normal, Gray zone or precancer), the softmax probability score (from 0.33 to 1) and the inference time will be displayed on the top left. The inference time represents the duration to run the steps 1 to 6 described above. The UI should look like this:

![image](https://user-images.githubusercontent.com/49137243/141835553-6ae9f9b3-1b34-4ef6-a0cd-0000478af618.png)


## Convert PyTorch model to a mobile optimized version
Python version: 3.8.0

Package requirements can be installed using the following CL:
```bash
pip install -r requirements.txt
```

The python script `mobile_model_conversion.py` includes the main steps to convert the PyTorch model into a version optimized and readable by an android app.
Currently, the script takes the ImageNet pretrained weights but the commented lines indicates the step to load the trained model. The script will generate the mobile optimized PyTorch version in the current directory `dummy_model.ptl`.

```bash
python mobile_model_conversion.py
```



