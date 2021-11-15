package org.pytorch.helloworld;

import android.content.Context;
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
import org.pytorch.MemoryFormat;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.lang.Math;
import java.nio.FloatBuffer;

import androidx.appcompat.app.AppCompatActivity;

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
      module = Module.load(assetFilePath(this, "dummy_model.ptl"));
    } catch (IOException e) {
      Log.e("CervicalApp", "Error reading assets", e);
      finish();
    }

    // showing image on UI
    ImageView imageView = findViewById(R.id.image);
    imageView.setImageBitmap(bitmap);


    // preparing input tensor
    final Tensor inputTensor = TensorImageUtils.bitmapToFloat32Tensor(bitmap,
            MEANNULL, STDONE);

    final Tensor outputTensor = module.forward(IValue.from(inputTensor)).toTensor();

    int mcIt = 0;
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
