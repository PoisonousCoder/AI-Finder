package com.example.aifinder;

import static android.app.PendingIntent.getActivity;

import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.media.ThumbnailUtils;
import android.os.Build;
import android.os.Bundle;
import android.provider.MediaStore;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import androidx.annotation.Nullable;
import androidx.annotation.RequiresApi;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.content.ContextCompat;

import com.example.aifinder.ml.ModelUnquant2;

import org.tensorflow.lite.DataType;

import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;


import android.Manifest;
import android.net.Uri;

public class MainActivity extends AppCompatActivity {
    Button camera, gallery;
    ImageView imageView;
    TextView result;

    int imageSize = 224;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        camera = findViewById(R.id.camerabutton);
        gallery = findViewById(R.id.gallerybutton);

        result = findViewById(R.id.result);
        imageView = findViewById(R.id.imageView);

        camera.setOnClickListener(new View.OnClickListener() {
            @RequiresApi(api = Build.VERSION_CODES.M)
            @Override
            public void onClick(View view) {
                if(checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED){
                    Intent cameraIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
                    startActivityForResult(cameraIntent, 3);
                }else{
                    requestPermissions(new String[]{Manifest.permission.CAMERA}, 100);
                }
            }
        });

        gallery.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Intent cameraIntent = new Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
               startActivityForResult(cameraIntent, 1);
            }
        });
    }

    public void classifyImage(Bitmap image){
        try {
            ModelUnquant2 model = ModelUnquant2.newInstance(getApplicationContext());

            // Creates inputs for reference.
            TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 224, 224, 3}, DataType.FLOAT32);
            ByteBuffer byteBuffer = ByteBuffer.allocateDirect(4 * imageSize * imageSize * 3);
            byteBuffer.order(ByteOrder.nativeOrder());

            // Resize and normalize the image
            Bitmap resizedImage = Bitmap.createScaledBitmap(image, imageSize, imageSize, false);

            int[] intValues = new int[imageSize * imageSize];
            resizedImage.getPixels(intValues, 0, imageSize, 0, 0, imageSize, imageSize);
            int pixel = 0;
            for (int i = 0; i < imageSize; i++) {
                for (int j = 0; j < imageSize; j++) {
                    int val = intValues[pixel++];
                    // Normalize the pixel values to the range [0, 1]
                    byteBuffer.putFloat(((val >> 16) & 0xFF) / 255.0f); // Red
                    byteBuffer.putFloat(((val >> 8) & 0xFF) / 255.0f);  // Green
                    byteBuffer.putFloat((val & 0xFF) / 255.0f);         // Blue
                }
            }

            inputFeature0.loadBuffer(byteBuffer);

            // Runs model inference and gets result.
            ModelUnquant2.Outputs outputs = model.process(inputFeature0);
            TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();

            // Extract confidence scores for AI and Human classes
            float[] confidence = outputFeature0.getFloatArray();

            float aiConfidence = confidence[0] * 100;    // Assuming class 0 is AI
            float humanConfidence = confidence[1] * 100; // Assuming class 1 is Human

            // Display confidence values
            result.setText(String.format("AI: %.2f%%, Human: %.2f%%", aiConfidence, humanConfidence));

            // Set the appropriate image in the resultImageView
            //if (aiConfidence > 50) {
               // resultImageView.setImageResource(R.drawable.robot_result);  // Set robot result image
            //} else if (humanConfidence > 50) {
                //resultImageView.setImageResource(R.drawable.human_result);  // Set human result image
            //} else if (aiConfidence == 50 && humanConfidence == 50) {
                //resultImageView.setImageResource(R.drawable.unsure_result);  // Set unsure result image
            //}

            // Releases model resources if no longer used.
            model.close();
        } catch (IOException e) {
            // Handle the exception
        }
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data){
        if(resultCode == RESULT_OK){
            if(requestCode == 3){
                Bitmap image = (Bitmap) data.getExtras().get("data");
                int dimension = Math.min(image.getWidth(), image.getHeight());
                image = ThumbnailUtils.extractThumbnail(image, dimension, dimension);
                imageView.setImageBitmap(image);
                image = Bitmap.createScaledBitmap(image, imageSize, imageSize, false);
                classifyImage(image);
            }else{
                Uri dat = data.getData();
                Bitmap image = null;
                try {
                    image = MediaStore.Images.Media.getBitmap(this.getContentResolver(), dat);
                } catch (IOException e) {
                    e.printStackTrace();
                }
                imageView.setImageBitmap(image);
                image = Bitmap.createScaledBitmap(image, imageSize, imageSize, false);
                classifyImage(image);
            }
        }
        super.onActivityResult(resultCode, resultCode, data);

    }
}
