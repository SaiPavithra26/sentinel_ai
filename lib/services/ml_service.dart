import 'dart:developer' as developer;
import 'package:tflite_flutter/tflite_flutter.dart';

/// A service that handles TensorFlow Lite model operations including loading,
/// running inference, error handling, and closing resources.
class MLService {
  Interpreter? _interpreter;
  bool _isModelLoaded = false;

  /// Returns whether the model has been successfully loaded.
  bool get isModelLoaded => _isModelLoaded;

  /// Loads the TFLite model from assets.
  /// 
  /// [ownerId] is the user ID corresponding to the model subdirectory.
  /// [customPath] allows specifying an alternative model path.
  Future<bool> loadModel({String ownerId = '1', String? customPath}) async {
    final modelPath = customPath ?? 'models/user_$ownerId/behavior_model.tflite';
    try {
      developer.log('MLService: Loading TFLite model from assets/$modelPath', name: 'MLService');
      
      // Configure options (using multiple threads for faster inference)
      final options = InterpreterOptions()..threads = 4;
      
      // Load interpreter from asset bundle
      _interpreter = await Interpreter.fromAsset(modelPath, options: options);
      
      // Allocate memory for input and output tensors
      _interpreter!.allocateTensors();
      
      _isModelLoaded = true;
      developer.log('MLService: Model loaded successfully.', name: 'MLService');
      return true;
    } catch (e, stackTrace) {
      _isModelLoaded = false;
      _interpreter = null;
      developer.log(
        'MLService: Error loading TFLite model from path: $modelPath. Error: $e',
        error: e,
        stackTrace: stackTrace,
        name: 'MLService',
      );
      return false;
    }
  }

  /// Runs inference on the provided feature vector and returns prediction confidence.
  /// 
  /// [featureVector] is the input vector containing preprocessed features.
  /// Returns a double representing prediction confidence, or 0.0 if an error occurs or the model isn't loaded.
  double runInference(List<double> featureVector) {
    if (!_isModelLoaded || _interpreter == null) {
      developer.log('MLService: Inference attempted but model is not loaded.', name: 'MLService');
      return 0.0;
    }

    try {
      // The model expects an input of shape [1, featureVector.length]
      final input = [featureVector];

      // The model yields an output of shape [1, 1] (probability of Owner)
      final output = List<List<double>>.generate(
        1,
        (_) => List<double>.filled(1, 0.0),
      );

      // Execute TFLite Inference
      _interpreter!.run(input, output);

      final confidence = output[0][0];
      return confidence;
    } catch (e, stackTrace) {
      developer.log(
        'MLService: Error running model inference. Error: $e',
        error: e,
        stackTrace: stackTrace,
        name: 'MLService',
      );
      return 0.0;
    }
  }

  /// Closes the interpreter session and releases native resources.
  void dispose() {
    try {
      if (_interpreter != null) {
        _interpreter!.close();
        _interpreter = null;
        developer.log('MLService: Interpreter session closed.', name: 'MLService');
      }
      _isModelLoaded = false;
    } catch (e) {
      developer.log('MLService: Error during disposal: $e', name: 'MLService');
    }
  }
}
