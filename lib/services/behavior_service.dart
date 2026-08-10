import 'dart:async';
import 'dart:developer' as developer;
import 'package:flutter/widgets.dart';
import '../models/behavior_data.dart';
import 'sensor_service.dart';
import 'tflite_service.dart';
import 'feature_extractor.dart';

/// Represents the result of a single behavioral authentication evaluation.
class AuthenticationResult {
  /// The probability confidence that the current user is the legitimate owner (from 0.0 to 1.0).
  final double confidence;

  /// Whether the user is authenticated as the owner based on the model's threshold.
  final bool isAuthenticated;

  /// The timestamp of when this evaluation occurred.
  final DateTime timestamp;

  /// The calibrated threshold used for this prediction.
  final double threshold;

  AuthenticationResult({
    required this.confidence,
    required this.isAuthenticated,
    required this.timestamp,
    required this.threshold,
  });

  @override
  String toString() {
    return 'AuthenticationResult(confidence: ${confidence.toStringAsFixed(4)}, '
        'isAuthenticated: $isAuthenticated, threshold: ${threshold.toStringAsFixed(4)})';
  }
}

/// A service that coordinates continuous behavioral authentication.
/// It monitors streams of sensor data, generates feature vectors at regular intervals,
/// evaluates them against the TensorFlow Lite model, and publishes results via a stream.
/// It also conforms to the application lifecycle to save power and resource allocations.
class BehaviorService with WidgetsBindingObserver {
  final SensorService _sensorService;
  final TfliteInferenceService _tfliteService;

  // Configuration settings
  final Duration _evaluationInterval;
  final int _maxBufferSize;

  // Internal states
  bool _isMonitoring = false;
  final List<BehaviorData> _dataBuffer = [];
  Timer? _evaluationTimer;
  StreamSubscription<BehaviorData>? _sensorSubscription;

  // Stream controller to emit continuous authentication results
  final StreamController<AuthenticationResult> _authResultStreamController =
      StreamController<AuthenticationResult>.broadcast();

  /// A broadcast stream of behavioral authentication results.
  Stream<AuthenticationResult> get authResultStream => _authResultStreamController.stream;

  /// Returns whether the continuous authentication monitoring is active.
  bool get isMonitoring => _isMonitoring;

  /// Returns a read-only list of the currently accumulated sensor data.
  List<BehaviorData> get currentBuffer => List.unmodifiable(_dataBuffer);

  /// Constructor for dependency injection of services and custom configuration.
  BehaviorService({
    required SensorService sensorService,
    required TfliteInferenceService tfliteService,
    Duration evaluationInterval = const Duration(seconds: 5),
    int maxBufferSize = 5000,
  })  : _sensorService = sensorService,
        _tfliteService = tfliteService,
        _evaluationInterval = evaluationInterval,
        _maxBufferSize = maxBufferSize;

  /// Starts the continuous behavioral authentication loop.
  void startMonitoring() {
    if (_isMonitoring) return;
    _isMonitoring = true;

    developer.log('BehaviorService: Starting continuous behavioral authentication...', name: 'BehaviorService');

    // Register this instance as an app lifecycle observer
    WidgetsBinding.instance.addObserver(this);

    // Start listening to the unified sensor data stream
    _startSensorSubscription();

    // Start the periodic evaluation timer
    _startEvaluationTimer();
  }

  /// Stops the continuous behavioral authentication loop and cleans up active resources.
  void stopMonitoring() {
    if (!_isMonitoring) return;
    _isMonitoring = false;

    developer.log('BehaviorService: Stopping continuous behavioral authentication...', name: 'BehaviorService');

    WidgetsBinding.instance.removeObserver(this);
    _stopSensorSubscription();
    _stopEvaluationTimer();
  }

  /// Starts the subscription to the sensor data stream.
  void _startSensorSubscription() {
    _stopSensorSubscription();
    
    _sensorSubscription = _sensorService.dataStream.listen(
      (BehaviorData data) {
        _addDataToBuffer(data);
      },
      onError: (error) {
        developer.log('BehaviorService: Sensor stream error received: $error', error: error, name: 'BehaviorService');
      },
    );
  }

  /// Cancels the sensor stream subscription.
  void _stopSensorSubscription() {
    _sensorSubscription?.cancel();
    _sensorSubscription = null;
  }

  /// Adds a new sensor snapshot to the FIFO sliding window buffer.
  void _addDataToBuffer(BehaviorData data) {
    if (_dataBuffer.length >= _maxBufferSize) {
      _dataBuffer.removeAt(0); // Evict oldest data point
    }
    _dataBuffer.add(data);
  }

  /// Starts the periodic timer that triggers TFLite model evaluation.
  void _startEvaluationTimer() {
    _stopEvaluationTimer();
    
    _evaluationTimer = Timer.periodic(_evaluationInterval, (timer) {
      _evaluateBehavior();
    });
  }

  /// Cancels the periodic evaluation timer.
  void _stopEvaluationTimer() {
    _evaluationTimer?.cancel();
    _evaluationTimer = null;
  }

  /// Extracts features from the current data buffer, runs inference, and posts the result.
  void _evaluateBehavior() {
    if (!_isMonitoring) return;

    if (!_tfliteService.isLoaded) {
      developer.log('BehaviorService: Evaluation skipped. Model is not loaded.', name: 'BehaviorService');
      return;
    }

    if (_dataBuffer.isEmpty) {
      developer.log('BehaviorService: Evaluation skipped. Data buffer is empty.', name: 'BehaviorService');
      return;
    }

    try {
      developer.log(
        'BehaviorService: Running evaluation on ${_dataBuffer.length} buffered points...',
        name: 'BehaviorService',
      );

      // 1. Generate feature vector using FeatureExtractor
      final rawFeatures = FeatureExtractor.extractFeatures(
        _dataBuffer,
        sessionDurationMins: 5.0, // baseline session window size
      );

      // 2. Run inference to get the confidence score (Owner probability)
      final confidence = _tfliteService.predict(rawFeatures);
      final isAuthenticated = confidence >= _tfliteService.threshold;

      // 3. Construct and emit the result
      final result = AuthenticationResult(
        confidence: confidence,
        isAuthenticated: isAuthenticated,
        timestamp: DateTime.now(),
        threshold: _tfliteService.threshold,
      );

      _authResultStreamController.add(result);
      developer.log('BehaviorService: Result emitted: $result', name: 'BehaviorService');
    } catch (e, stackTrace) {
      developer.log(
        'BehaviorService: Error during behavior evaluation: $e',
        error: e,
        stackTrace: stackTrace,
        name: 'BehaviorService',
      );
    }
  }

  /// Clears the accumulated data buffer.
  void clearBuffer() {
    _dataBuffer.clear();
    developer.log('BehaviorService: Data buffer cleared.', name: 'BehaviorService');
  }

  // --- AppLifecycleState Hooks for Power/Battery Optimization ---

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    if (!_isMonitoring) return;

    switch (state) {
      case AppLifecycleState.paused:
      case AppLifecycleState.inactive:
        // Pause resource-intensive operations when backgrounded
        _stopEvaluationTimer();
        _stopSensorSubscription();
        developer.log('BehaviorService: App backgrounded. Suspended authentication loop.', name: 'BehaviorService');
        break;
      case AppLifecycleState.resumed:
        // Restore operations when returning to foreground
        _startSensorSubscription();
        _startEvaluationTimer();
        developer.log('BehaviorService: App foregrounded. Restored authentication loop.', name: 'BehaviorService');
        break;
      case AppLifecycleState.detached:
      case AppLifecycleState.hidden:
        _stopEvaluationTimer();
        _stopSensorSubscription();
        break;
    }
  }

  /// Disposes of the service, closing open streams.
  void dispose() {
    stopMonitoring();
    _authResultStreamController.close();
    developer.log('BehaviorService: Service disposed.', name: 'BehaviorService');
  }
}
