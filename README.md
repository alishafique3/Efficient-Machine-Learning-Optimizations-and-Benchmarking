# Efficient Machine Learning: Optimizations and Benchmarking
Deep learning is becoming popular nowadays. It is being used in different applications like classification, segmentation, pose estimation, augmented reality and self-driving cars. Primary goal in deep learning applications is to achieve accuracy. This can be done using big models but these complex models leads several issues in real time applications. These real-time applications run on edge devices but large models put constraints on memory and latency.

TensorFlow Lite benchmark tools currently measure and calculate statistics for the following important performance metrics:
* Initialization time
* Inference time of warmup state
* Inference time of steady state
* Memory usage during initialization time
* Overall memory usage


## Methodology
Base code for model training and dataset is provided in the blog at Pyimagesearch https://www.pyimagesearch.com/2019/11/18/fire-and-smoke-detection-with-keras-and-deep-learning/ by Adrian Rosebrock on November 18, 2019. 
This project has contributed to the following: 
1.	Model training is modified for pruning, clustering, and collaborative optimizations.
2.	Post-Training optimization
3.	Benchmarking in Android devices

![Picture4](https://github.com/alishafique3/Efficient-Machine-Learning_Optimizations-and-Benchmarking/assets/17300597/944f3821-2b1a-4b18-b820-ff8909bc8561)



### Step1: Loading and preprocessing of the dataset
A dataset can be accessed either by uploading it in the drive of google colab or google drive. In this project, the dataset has been uploaded on google drive and it is accessed in the code by mounting google drive in Colab. 

### Step2: Model architecture and training
A deep neural network is used for classification. It contains convolutional layers, max pooling, and batch normalization layers. Adam optimizer is used for training with binary cross entropy loss function. It is trained for 50 epochs. Four different models are trained with different optimizations.

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
After fine-tunning we need to remove clustering tf.variables that are needed for clustering otherwise it will increase the model size.
```python
stripped_pruned_model = tfmot.sparsity.keras.strip_pruning(pruned_model)
```
After clustering we will again apply post-training quantization to get quantized model.
```python
converter = tf.lite.TFLiteConverter.from_keras_model(stripped_pruned_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
pqat_tflite_model = converter.convert()
```







