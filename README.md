# Efficient Machine Learning Inference: Optimizations and Benchmarking
Deep learning is becoming popular nowadays. It is being used in different applications such as classification, segmentation, pose estimation, augmented reality and self-driving cars. The primary goal in deep learning applications is to achieve accuracy. This can be accomplished using big models but these complex models give rise to several issues in real-time applications. These real-time applications run on edge devices have limited memory and computation resources that results in reduction of model inference performance. 

Model inference can be improved using optimization techniques such as pruning, clustering and quantization. Optimized models enable efficient use of memory and make computations simple, thereby resulting in the following advantages

1. **Memory Usage:** using Integer or low-bits bits for input, weights, activations, and output give rise to less use of memory
2. **Power Consumption:** less memory access and simpler computation reduce power consumption significantly
3. **Latency:** less memory access and simpler computation also speed up the inference
4. **Silicon Area:** Integer or low-bits require less silicon area for computational hardware as compared to floating bits

In this project, tensorflow keras is used to develop and train CNN model for classifiying fire and non-fire dataset. Model is optimized using tensorflow model optimization library. Optimized model is then converted into Tensorlite format. The performance of Tensorlite model is bechmarked on Android device using TensorFlow Lite benchmark tools. These tools measure several important performance metrics:
* Initialization time
* Inference time of warmup state
* Inference time of steady state
* Memory usage during initialization time
* Overall memory usage

## Description of Optimization
### Magnitude-based weight pruning
In magnitude-based weight pruning, model weights, having values less than a threshold, are made zero during the training process. It develops sparsity in the model which helps in model compression. We can also skip those zero weights during inference resulting in an improvement in latency. In unstructured pruning, individual weight connections are removed from a network by setting them to zero. In structured pruning, groups of weight connections are removed together, such as entire channels or filters. Unfortunately, structured pruning severely limits the maximum sparsity that limits both performance and memory improvements.

![Picture8](https://github.com/alishafique3/Efficient-Machine-Learning_Optimizations-and-Benchmarking/assets/17300597/e4fe8f1c-6cdc-468d-b14d-af51bf0206da)


### Clustering
Clustering, also called weight sharing, helps to make models more memory-efficient by reducing the number of different weight values. In this process, the weights of each layer are grouped into clusters. Within each cluster, all model weights share the same value which is known as the centroid value of the cluster.
![Picture7](https://github.com/alishafique3/Efficient-Machine-Learning_Optimizations-and-Benchmarking/assets/17300597/49b43d76-5dbf-44e2-a9e9-a9fdfba8f065)



### Quantization
In Quantization, precision of the model weights, activation, input and output is decreased by reducing the the number of bits representing numerical values. Using lower precision such as FP16 or INT8 as compared to FP32, makes the model memory-efficient and helps in faster execution. In this technique, high-precision values are mapped into lower-precision values using quantization-aware training, post-training quantization or hybrid quantization which is combination of both. Quantization is helpful for deploying models in resource-constraied edge devices as it reduces computational and memory requirements with acceptable accuracy.
![Picture6](https://github.com/alishafique3/Efficient-Machine-Learning_Optimizations-and-Benchmarking/assets/17300597/6f1fb35e-e79e-4d70-9163-393726c12939)



## Methodology
The base code for model training and dataset is provided in the [blog](https://www.pyimagesearch.com/2019/11/18/fire-and-smoke-detection-with-keras-and-deep-learning/) at Pyimagesearch by Adrian Rosebrock. 
This project has contributed to the following: 
1.	Model training is modified for pruning, clustering, and collaborative optimizations.
2.	Post-Training optimization
3.	Benchmarking in Android device

![Picture4](https://github.com/alishafique3/Efficient-Machine-Learning_Optimizations-and-Benchmarking/assets/17300597/944f3821-2b1a-4b18-b820-ff8909bc8561)



### Step1: Loading and preprocessing of the dataset
A dataset contains images of two calsses (NonFire and Fire). The total size of the dataset is 4008 images and it is split up into training dataset and validation dataset. Following are the hyperparameters used for regular model training.
| Hyperparameters        | Value           |
|:-------------:|:-------------:|
| Dataset access      | Google Drive |
| Classes      | 2 (nonFire,Fire)|
| Batch Size | 64      |
| No. of Epochs | 50      |
| Train dataset | (3006,128,128,3)      |
| Validation dataset | (1002,128,128,3)      |
| Preprocessing | Resize, normalization, one-hot encoding and split      |


### Step2: Model architecture and training
A deep neural network is used for classification. It contains convolutional layers, max pooling layers, and batch normalization layers. Adam optimizer is used for training with binary cross entropy loss function. It is trained for 50 epochs. Four different models are trained with different optimizations. The model architecture is given below

![Picture5](https://github.com/alishafique3/Efficient-Machine-Learning_Optimizations-and-Benchmarking/assets/17300597/3b65816f-2b3a-4770-ba23-3832e8f8ee7f)

#### Unoptimized Baseline Model:
In this model, training is done with no optimization. This model is used to compare with optimized versions. This model is converted into TensorLite format. 
Baseline model is developed using following code:
```python
converter1 = tf.lite.TFLiteConverter.from_keras_model(model)
baseline_tflite_model = converter1.convert()
```

#### Dynamic range quantization Model:
In this model, training is done without any optimization, while model is converted with dynamic range quantization during tensorlite conversion process. 
```python
converter2 = tf.lite.TFLiteConverter.from_keras_model(model)
converter2.optimizations = [tf.lite.Optimize.DEFAULT]
quantized_tflite_model = converter2.convert()
```

#### Float16 quantization Model:
In this model, training is done without any optimization, while model is converted with float16 quantization during tensorlite conversion process. 
```python
converter3 = tf.lite.TFLiteConverter.from_keras_model(model)
converter3.optimizations = [tf.lite.Optimize.DEFAULT]
converter3.target_spec.supported_types = [tf.float16]
fquantized_tflite_model = converter3.convert()
```

#### Clustered and Quantized Model:
In this model, training is done with regular model then fine tuning is done with clustering optimization. In the end, the model is converted with dynamic range quantization during tensorlite conversion process.

```python
#Fine Tunning after regular training
cluster_weights = tfmot.clustering.keras.cluster_weights
CentroidInitialization = tfmot.clustering.keras.CentroidInitialization
clustering_params = {
  'number_of_clusters': 8,
  'cluster_centroids_init': CentroidInitialization.KMEANS_PLUS_PLUS,
  'cluster_per_channel': True,
}
clustered_model = cluster_weights(model, **clustering_params)
# Use smaller learning rate for fine-tuning
opt = tf.keras.optimizers.Adam(learning_rate=1e-5)
clustered_model.compile(loss="binary_crossentropy", optimizer=opt,
  metrics=["accuracy"])
clustered_model.summary()
```
After Fine-tuning: 

After fine-tunning we need to remove clustering tf.variables that are needed for clustering otherwise it will increase the model size.
```python 
stripped_clustered_model = tfmot.clustering.keras.strip_clustering(clustered_model)
```
After clustering we will again apply post-training quantization to get quantized model.
```python
Converter4 = tf.lite.TFLiteConverter.from_keras_model(stripped_clustered_model)
Converter4.optimizations = [tf.lite.Optimize.DEFAULT]
quantized_tflite_model = converter4.convert()
```
#### Pruning and Quantized Model
In this model, training is done with regular model then fine tuning is done with pruning optimization. In the end, the model is converted with dynamic range quantization during tensorlite conversion process. 
After Training, Fine Tuning stage:
```python
prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
pruning_params = {
      'pruning_schedule': tfmot.sparsity.keras.ConstantSparsity(0.5, begin_step=0, frequency=100)
  }
callbacks = [
  tfmot.sparsity.keras.UpdatePruningStep()
]

pruned_model = prune_low_magnitude(model, **pruning_params)

# Use smaller learning rate for fine-tuning
opt = tf.keras.optimizers.Adam(learning_rate=1e-5)

pruned_model.compile(loss="binary_crossentropy", optimizer=opt,
  metrics=["accuracy"])
 ```
After fine-tunning we need to remove pruning tf.variables that are needed for clustering otherwise it will increase the model size.
```python
stripped_pruned_model = tfmot.sparsity.keras.strip_pruning(pruned_model)
```
After clustering we will again apply post-training quantization to get quantized model.
```python
converter = tf.lite.TFLiteConverter.from_keras_model(stripped_pruned_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
pqat_tflite_model = converter.convert()
```

### Step3: Benchmarking on Android Device
Android Debugging Bridge(ADB) can be installed in Windows and Linux. I have found these videos very useful to setup [Link](https://www.youtube.com/watch?v=26GI3z6tI3E). and get familiarized with ADB [Link](https://www.youtube.com/watch?v=uOPcUjVl2YQ). Once ADB is set up, mobile can be controlled via computer with ADB commands using USB or wifi connectivity. Remember android devices should be in USB Debugging mode or Wireless debugging mode with USB or wifi connection respectively. Android emulator can also be used for benchmarking the models. Android emulator can be found using android studio. (Android Studio contains both an emulator and ADB).
`adb devices` will list all Android devices/emulators connected to the computer. If it does not work or if it shows offline, try to connect Android again by removing the ADB connection using `adb kill-server` and try again using `adb devices`

Once the Android device connection with the computer is established with ADB, benchmarking files and target TFLite models will be sent to the Android device using ADB commands. There are two options for using the benchmark tool with Android. 
1.	Native benchmark binary 
2.	Android benchmark app, a better tool to measure how the model would perform in the app.

Both benchmarking files are available in android_aarch64 and android_arm at [Link](https://www.tensorflow.org/lite/performance/measurement). To find your Android device-specific architecture and system details, you can download a droid hardware app or similar app from Play Store.
#### Android benchmark app 
once Android benchmarking file is downloaded, keep TFLite models and Android benchmark app in same folder and open terminal in that folder, start following lines
```python
adb devices
adb install -r -d -g android_aarch64_benchmark_model.apk  # for android_aarch64 benchmarking  file
adb push mobilenet.tflite /data/local/tmp # for model  file
adb shell am start -S \-n org.tensorflow.lite.benchmark/.BenchmarkModelActivity \--es args '"--graph=/data/local/tmp/mobilenet.tflite \--num_threads=4"'
adb logcat | findstr "Inference timings"
```

#### Native benchmark binary
once Android benchmarking file is downloaded, keep TFLite models and Android benchmark app in same folder and open terminal in that folder, start following lines
```python
adb push android_aarch64_benchmark_model /data/local/tmp # for android_aarch64 benchmarking  file
adb shell chmod +x /data/local/tmp/android_aarch64_benchmark_model
adb push mobilenet.tflite /data/local/tmp # for model  file
adb shell /data/local/tmp/android_aarch64_benchmark_model \--graph=/data/local/tmp/mobilenet.tflite \--num_threads=4
```

graph is a required parameter.
*	graph: string 
The path to the TFLite model file.
You can specify more optional parameters for running the benchmark.
*	`num_threads`: int (default=1) The number of threads to use for running TFLite interpreter. A thread is a virtual component that handles the tasks of a CPU core. Multithreading is the ability of the CPU to divide up the work among multiple threads instead of giving it to a single core, to enable concurrent processing. The multiple threads are processed by the different CPU cores in parallel, to speed up performance and save time.
*	`use_gpu`: bool (default=false) you can enable use of GPU-accelerated execution of your models using a delegate. Delegates act as hardware drivers for TensorFlow Lite, allowing you to run the code of your model on GPU processors.
*	`use_nnapi`: bool (default=false) Use Android Neural Networks API (NNAPI) delegate.  It provides acceleration for TensorFlow Lite models on Android devices with supported hardware accelerators including, Graphics Processing Unit (GPU), Digital Signal Processor (DSP) and Neural Processing Unit (NPU).
*	`use_xnnpack`: bool (default=false) Use XNNPACK delegate. XNNPACK is a highly optimized library of neural network inference operators for ARM, x86, and WebAssembly architectures in Android, iOS, Windows, Linux, macOS, and Emscripten environments.
*	`use_hexagon`: bool (default=false) Use Hexagon delegate. This delegate leverages the Qualcomm Hexagon library to execute quantized kernels on the DSP. Note that the delegate is intended to complement NNAPI functionality, particularly for devices where NNAPI DSP acceleration is unavailable

## Result
Android Device use for this project is Xiaomi Mi A2 with octacore processor and Adreno512 GPU. During benchmarking, 4 CPU threads are used. Runtime memory and model size are in MB while inference time is an average time in microseconds. 
| Optimization Technique        | Size           | InferTime_CPU  | Runtime_Memory_CPU  | InferTime_GPU  | Runtime_Memory_GPU  | InferTime_NNAPI  | Runtime_Memory_NNAPI  |
:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|
| Base_Model No Optimization      | 8.5 | 6719.84 | 11.4      | 5175.1 | 49.4 | 9382      | 11.46 |
| Dynamic Range Quantized      | 2.14 | 8375 | 5.7      | 5075 | 43.34 | 8307      | 5.75 |
| Float16 Quantized      | 4.26 | 7563 | 15.69      | 5173 | 45.34 | 6553      | 16.05 |
| Clustered and Quantized      | 2.14 | 8897 | 5.98      | 5214 | 43.35 | 8491      | 5.9 |
| Pruned and Quantized      | 2.14 | 8032 | 5.88      | 5057.1 | 43.32 | 7483      | 6.02 |

## Conclusion
In this project, different optimized models have been compared on android device. Dynamic quantization plays remarkably well among these optimized models. This project can be extended on different datasets, models and hardware to see the performance of optimization techniques.

## References
1.	Pyimagesearch Website: Fire and smoke detection with Keras and deep learning link: https://www.pyimagesearch.com/2019/11/18/fire-and-smoke-detection-with-keras-and-deep-learning/
2.	Youtube Website: tinyML Talks: A Practical guide to neural network quantization link: https://www.youtube.com/watch?v=KASuxB3XoYQ 
3.	Medium Website: Neural Network Compression using Quantization link: https://medium.com/sharechat-techbyte/neural-network-compression-using-quantization-328d22e8855d
4.	Tensorflow Website: Performance Measurement link: https://www.tensorflow.org/lite/performance/measurement
5.	Tensorflow Website: Post Training Quantization link: https://www.tensorflow.org/lite/performance/post_training_quantization#dynamic_range_quantization
6.	Tensorflow Website: Model Optimization Pruning link: https://www.tensorflow.org/model_optimization/guide/pruning
7.	Tensorflow Website: Weight clustering link: https://blog.tensorflow.org/2020/08/tensorflow-model-optimization-toolkit-weight-clustering-api.html
8.	Tensorflow Website: pruning link: https://blog.tensorflow.org/2019/05/tf-model-optimization-toolkit-pruning-API.html
9.	Tensorflow Website: Quantization aware training optimization link: https://blog.tensorflow.org/2020/04/quantization-aware-training-with-tensorflow-model-optimization-toolkit.html




