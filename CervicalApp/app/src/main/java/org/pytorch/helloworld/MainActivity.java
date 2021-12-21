package org.pytorch.helloworld;

import android.content.res.AssetManager;
import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.util.Log;
import android.widget.ImageView;
import android.widget.TextView;
import android.media.Image;
import android.media.ImageReader;

import org.pytorch.IValue;
import org.pytorch.Module;
import org.pytorch.Tensor;
import org.pytorch.torchvision.TensorImageUtils;
import org.tensorflow.lite.nnapi.NnApiDelegate;
import org.tensorflow.lite.HexagonDelegate;
import org.pytorch.Device;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.lang.Math;
import java.nio.MappedByteBuffer;
import java.io.FileInputStream;
import java.nio.channels.FileChannel;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Arrays;


import androidx.appcompat.app.AppCompatActivity;

//import ai.onnxruntime.OrtEnvironment;
//import ai.onnxruntime.OrtException;
//import ai.onnxruntime.OrtSession;

import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.gpu.CompatibilityList;
import org.tensorflow.lite.gpu.GpuDelegate;
import org.tensorflow.lite.flex.FlexDelegate;

public class MainActivity extends AppCompatActivity {
  public static float[] MEANNULL = new float[] {0.0f, 0.0f, 0.0f};
  public static float[] STDONE = new float[] {1.0f, 1.0f, 1.0f};

  public static float[] add(float[] first, float[] second) {
    int length = Math.min(first.length, second.length);
    float[] result = new float[length];
    for (int i = 0; i < length; i++) {
      result[i] = first[i] + second[i];
    }
    return result;
  }

  @Override
  protected void onCreate(Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);
    setContentView(R.layout.activity_main);

    Bitmap bitmap = null;
    Module module = null;
    long start = System.currentTimeMillis();

    try {
      // creating bitmap from packaged into app android asset 'image.jpg',
      // app/src/main/assets/image.jpg
      bitmap = BitmapFactory.decodeStream(getAssets().open("dummy_img.jpg"));
      // Crop image with bounding box coordinate
      bitmap = Bitmap.createBitmap(bitmap, 708, 64, 1322, 1719);
      // Resize image
      bitmap = Bitmap.createScaledBitmap(bitmap, 256, 256, true);

      // loading serialized torchscript module from packaged into app android asset model.pt,
      // app/src/model/assets/model.pt

      // cpu
      // module = Module.load(assetFilePath(this, "dummy_model.ptl"));
      // module = Module.load(assetFilePath(this, "dummy_model_quantized.ptl"));
      // module = Module.load(assetFilePath(this, "dummy_model_resnet.ptl"));
      // module = Module.load(assetFilePath(this, "dummy_model_resnet.ptl"));
      // module = Module.load(assetFilePath(this, "dummy_model_densenet_nnapi.ptl"));

      // onnx - not working yet
      // OrtSession.SessionOptions session_options = new OrtSession.SessionOptions();
      // session_options.addConfigEntry("session.load_model_format", "ORT");
      //
      // OrtEnvironment env = OrtEnvironment.getEnvironment();
      // OrtSession session = env.createSession(assetFilePath(this,"model.basic.ort"), session_options);

      // vulkan - not working yet
      // module = Module.load ( assetFilePath(this,"dummy_model_vulkan.pt"), null, Device.VULKAN);


    } catch (IOException e) {
      Log.e("CervicalApp", "Error reading assets", e);
      finish();
    }

    // showing image on UI
    ImageView imageView = findViewById(R.id.image);
    imageView.setImageBitmap(bitmap);

    int mcIt = 50;
    // Run PyTorch
    // float [] softmaxScores = runPytorchInference(bitmap, module, mcIt)

    // Run TFlite
    float [] softmaxScores = runTFInference(bitmap, mcIt);

    // searching for the index with maximum score
    float maxScore = -Float.MAX_VALUE;
    int maxScoreIdx = -1;
    for (int i = 0; i < softmaxScores.length; i++) {
      System.out.println(softmaxScores[i]);
      if (softmaxScores[i] > maxScore) {
        maxScore = softmaxScores[i];
        maxScoreIdx = i;
      }
    }

    String className = CervixClass.CERVIX_CLASSES[maxScoreIdx];
    long end = System.currentTimeMillis();
    long elapsedTime = (end - start);

    // showing className on UI
    TextView textView = findViewById(R.id.text);
    String classNameScore = className + " - score:" + maxScore + " - Inference duration: " + elapsedTime + " ms";
    textView.setText(classNameScore);
  }

  /**
   * Copies specified asset to the file in /files app directory and returns this file absolute path.
   *
   * @return absolute file path
   */
  public static String assetFilePath(Context context, String assetName) throws IOException {
    File file = new File(context.getFilesDir(), assetName);
    if (file.exists() && file.length() > 0) {
      return file.getAbsolutePath();
    }

    try (InputStream is = context.getAssets().open(assetName)) {
      try (OutputStream os = new FileOutputStream(file)) {
        byte[] buffer = new byte[4 * 1024];
        int read;
        while ((read = is.read(buffer)) != -1) {
          os.write(buffer, 0, read);
        }
        os.flush();
      }
      return file.getAbsolutePath();
    }
  }

  private float [] runTFInference(Bitmap bitmap, int mcIt) {
    Interpreter interpreter = null;

    try {
      interpreter = getTFInterpreter("model.tflite");

    } catch (IOException e) {
      e.printStackTrace();
    }

    final int nOutputNeurons = 3;
    float [][] result = new float[1][nOutputNeurons];
    assert interpreter != null;
    ByteBuffer byteBuffer = convertBitmapToByteBuffer(bitmap);
    interpreter.run(byteBuffer, result);

    float[] softmaxScores = new float[nOutputNeurons];

    if (mcIt > 0) {
      // Multiple forward passes to simulate MC iterations
      float[] mcScore = new float[nOutputNeurons];
      for (int i = 0; i < mcIt; i++) {
        interpreter.run(byteBuffer, result);
        mcScore = add(mcScore, result[0]);
      }
      // Get mean score
      for (int i = 0; i < nOutputNeurons; i++) {
        softmaxScores[i] = mcScore[i] / mcIt;
      }
    }
    else {
      softmaxScores = result[0];
    }

    return softmaxScores;
  }

  private float [] runPytorchInference(Bitmap bitmap, Module module, int mcIt) {
    // preparing input tensor
    final Tensor inputTensor = TensorImageUtils.bitmapToFloat32Tensor(bitmap,
            MEANNULL, STDONE);

    final Tensor outputTensor = module.forward(IValue.from(inputTensor)).toTensor();

    final int nOutputNeurons = 3;
    float[] softmaxScores = new float[nOutputNeurons];

    if (mcIt > 0) {
      // Multiple forward passes to simulate MC iterations
      float[] mcScore = new float[nOutputNeurons];
      for (int i = 0; i < mcIt; i++) {
        float[] score = module.forward(IValue.from(inputTensor)).toTensor().getDataAsFloatArray();
        softmaxScores = softmax(score);
        mcScore = add(mcScore, softmaxScores);
      }

      // Get mean score
      for (int i = 0; i < nOutputNeurons; i++) {
        softmaxScores[i] = mcScore[i] / mcIt;
      }
    }
    else {
      // getting tensor content as java array of floats
      final float[] scores = outputTensor.getDataAsFloatArray();
      softmaxScores = softmax(scores);
    }

    return softmaxScores;
  }

  private ByteBuffer convertBitmapToByteBuffer(Bitmap bitmap) {
    ByteBuffer byteBuffer;
    int BATCH_SIZE = 1;
    int PIXEL_SIZE = 3;
    int inputSize = 256;
    boolean quant = false;
    int IMAGE_MEAN = 0;
    int IMAGE_STD = 1;

    byteBuffer = ByteBuffer.allocateDirect(4 * BATCH_SIZE * inputSize * inputSize * PIXEL_SIZE);

    byteBuffer.order(ByteOrder.nativeOrder());
    int[] intValues = new int[inputSize * inputSize];
    bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
    int pixel = 0;
    for (int i = 0; i < inputSize; ++i) {
      for (int j = 0; j < inputSize; ++j) {
        final int val = intValues[pixel++];
        byteBuffer.putFloat((((val >> 16) & 0xFF)-IMAGE_MEAN)/IMAGE_STD);
        byteBuffer.putFloat((((val >> 8) & 0xFF)-IMAGE_MEAN)/IMAGE_STD);
        byteBuffer.putFloat((((val) & 0xFF)-IMAGE_MEAN)/IMAGE_STD);

      }
    }
    return byteBuffer;
  }


  private MappedByteBuffer loadModelFile(Context context, String modelPath) throws IOException {
    AssetFileDescriptor fileDescriptor = context.getAssets().openFd(modelPath);
    FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
    FileChannel fileChannel = inputStream.getChannel();
    long startOffset = fileDescriptor.getStartOffset();
    long declaredLength = fileDescriptor.getDeclaredLength();
    return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
  }

  public Interpreter getTFInterpreter(String modelPath) throws IOException {
    Interpreter.Options options = new Interpreter.Options();

    // Use NNAPI
    NnApiDelegate nnApiDelegate = new NnApiDelegate();
    options.addDelegate(nnApiDelegate);

    // Use GPU, comment if CPU wanted
//    CompatibilityList compatList = new CompatibilityList();
//    GpuDelegate.Options delegateOptions = compatList.getBestOptionsForThisDevice();
//    GpuDelegate gpuDelegate = new GpuDelegate(delegateOptions);
//    options.addDelegate(gpuDelegate);

    // USE CPU, comment if GPU wanted
//    options.setNumThreads(Runtime.getRuntime().availableProcessors());

    return new Interpreter(loadModelFile(this, modelPath), options);
  }

  public float[] softmax(float[] input) {
    float[] exp = new float[input.length];
    float sum = 0;
    for(int neuron = 0; neuron < exp.length; neuron++) {
      double expValue = Math.exp(input[neuron]);
      exp[neuron] = (float)expValue;
      sum += exp[neuron];
    }

    float[] output = new float[input.length];
    for(int neuron = 0; neuron < output.length; neuron++) {
      output[neuron] = exp[neuron] / sum;
    }

    return output;
  }
}