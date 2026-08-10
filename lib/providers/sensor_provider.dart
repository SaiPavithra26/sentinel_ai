import 'dart:async';
import 'package:flutter/foundation.dart';
import 'package:sensors_plus/sensors_plus.dart';
import 'package:permission_handler/permission_handler.dart';
import '../models/behavior_data.dart';
import '../services/sensor_service.dart';
import '../services/tflite_service.dart';
import '../services/behavior_service.dart';

/// A production-ready State Management Provider that handles permissions,
/// aggregates gesture events, manages rolling interaction frequency,
/// maintains an in-memory buffer of data points, and runs real-time TFLite inference.
class SensorProvider with ChangeNotifier {
  final SensorService _sensorService = SensorService();
  final TfliteInferenceService _tfliteService = TfliteInferenceService();
  late final BehaviorService _behaviorService;
  StreamSubscription<BehaviorData>? _sensorSubscription;

  // In-memory data storage
  final List<BehaviorData> _collectedData = [];
  
  /// Maximum number of records held in memory to prevent excessive memory usage.
  static const int maxBufferSize = 5000;

  // Operational states
  bool _isMonitoring = false;
  bool _hasPermissions = false;
  
  // Real-time Inference states
  double _ownerProbability = 0.0;
  bool _isAuthenticated = false;
  String _activeOwnerId = '1';
  
  // Interaction tracking (rolling 60-second window)
  final List<DateTime> _interactionTimestamps = [];
  static const Duration _frequencyWindow = Duration(seconds: 60);

  // Keystroke metrics variables
  DateTime? _lastKeystrokeTime;
  int _keystrokeCount = 0;
  DateTime? _typingSessionStart;

  // --- Getters ---
  
  /// Read-only snapshot of the stored behavioral data records.
  List<BehaviorData> get collectedData => List.unmodifiable(_collectedData);
  
  /// The active Owner ID targeted for real-time authentication.
  String get activeOwnerId => _activeOwnerId;
  
  /// True if active data logging is underway.
  bool get isMonitoring => _isMonitoring;
  
  /// True if required sensor permissions have been granted.
  bool get hasPermissions => _hasPermissions;
  
  /// Total records in the current memory buffer.
  int get dataCount => _collectedData.length;

  /// Current predicted probability of the owner.
  double get ownerProbability => _ownerProbability;

  /// True if the user is authenticated as the legitimate owner.
  bool get isAuthenticated => _isAuthenticated;

  /// Threshold used to authenticate.
  double get threshold => _tfliteService.threshold;

  /// Equal Error Rate of the loaded model.
  double get eer => _tfliteService.eer;

  /// True if the TFLite model is successfully loaded.
  bool get isModelLoaded => _tfliteService.isLoaded;

  /// Calculates the interaction count in a sliding 60-second window.
  double get interactionFrequency {
    _cleanInteractionTimestamps();
    return _interactionTimestamps.length.toDouble();
  }

  /// Calculates typing speed in Words Per Minute (WPM) dynamically.
  double get typingSpeedWpm {
    if (_typingSessionStart == null || _keystrokeCount == 0) return 0.0;
    final durationMin = DateTime.now().difference(_typingSessionStart!).inMilliseconds / 60000.0;
    if (durationMin <= 0.0) return 0.0;
    // Standard estimation: (keystroke_count / 5) / duration_in_minutes
    return (_keystrokeCount / 5.0) / durationMin;
  }

  SensorProvider() {
    _checkPermissionStatus();
    _behaviorService = BehaviorService(
      sensorService: _sensorService,
      tfliteService: _tfliteService,
      evaluationInterval: const Duration(seconds: 5),
    );
    // Listen to continuous behavioral authentication stream results
    _behaviorService.authResultStream.listen((result) {
      _ownerProbability = result.confidence;
      _isAuthenticated = result.isAuthenticated;
      notifyListeners();
    });
  }

  /// Updates local permission state representation.
  Future<void> _checkPermissionStatus() async {
    final activityStatus = await Permission.activityRecognition.status;
    final sensorsStatus = await Permission.sensors.status;
    _hasPermissions = activityStatus.isGranted && sensorsStatus.isGranted;
    notifyListeners();
  }

  /// Request required runtime permissions (sensors + activity recognition).
  Future<bool> requestPermissions() async {
    final statuses = await [
      Permission.sensors,
      Permission.activityRecognition,
    ].request();

    final sensorsGranted = statuses[Permission.sensors]?.isGranted ?? false;
    final activityGranted = statuses[Permission.activityRecognition]?.isGranted ?? false;

    _hasPermissions = sensorsGranted && activityGranted;
    notifyListeners();
    return _hasPermissions;
  }

  /// Starts the continuous collection session.
  /// 
  /// Returns `true` if successfully started.
  Future<bool> startCollection({
    Duration sensorInterval = SensorInterval.normalInterval,
    Duration? samplingPeriod,
    String ownerId = '1',
  }) async {
    if (!_hasPermissions) {
      final granted = await requestPermissions();
      if (!granted) {
        debugPrint('SensorProvider: Access denied. Cannot start data logging.');
        return false;
      }
    }

    if (_isMonitoring) return true;

    _activeOwnerId = ownerId;
    
    // Initialize TFLite service
    await _tfliteService.initialize(ownerId: ownerId);

    _sensorService.startMonitoring(
      sensorInterval: sensorInterval,
      samplingPeriod: samplingPeriod,
    );

    // Cancel existing stream handlers to avoid double allocation
    await _sensorSubscription?.cancel();
    
    // Subscribe to unified physical and logical sensor data stream
    _sensorSubscription = _sensorService.dataStream.listen((data) {
      _addDataPoint(data);
    });

    // Start behavioral authentication engine
    _behaviorService.startMonitoring();

    _isMonitoring = true;
    notifyListeners();
    return true;
  }

  /// Stops the current collection session and cleans up subscriptions.
  Future<void> stopCollection() async {
    if (!_isMonitoring) return;

    _sensorService.stopMonitoring();
    await _sensorSubscription?.cancel();
    _sensorSubscription = null;

    _isMonitoring = false;
    _resetTypingSession();
    
    // Reset evaluation values
    _ownerProbability = 0.0;
    _isAuthenticated = false;
    
    _tfliteService.dispose();
    _behaviorService.stopMonitoring();
    _behaviorService.clearBuffer();
    notifyListeners();
  }

  /// Registers a tap interaction at (x, y) with its specific holding duration.
  void registerTap({
    required double x,
    required double y,
    required int durationMs,
  }) {
    _logInteractionTimestamp();
    
    if (_isMonitoring) {
      _sensorService.recordInteraction(
        touchX: x,
        touchY: y,
        tapDurationMs: durationMs,
        interactionFrequency: interactionFrequency,
      );
    }
  }

  /// Registers a swipe gesture, logging coordinate, size, speed, and time metrics.
  void registerSwipe({
    required double startX,
    required double startY,
    required double distance,
    required String direction,
    required double speed,
    required int durationMs,
  }) {
    _logInteractionTimestamp();

    if (_isMonitoring) {
      _sensorService.recordInteraction(
        touchX: startX,
        touchY: startY,
        swipeDistance: distance,
        swipeDirection: direction,
        swipeSpeed: speed,
        swipeDurationMs: durationMs,
        interactionFrequency: interactionFrequency,
      );
    }
  }

  /// Registers a typing keyboard action to measure latency and typing rates.
  void registerKeystroke() {
    final now = DateTime.now();
    _logInteractionTimestamp();

    int? intervalMs;
    if (_lastKeystrokeTime != null) {
      intervalMs = now.difference(_lastKeystrokeTime!).inMilliseconds;
      // Reset sessions if delay exceeds 5 seconds to ensure clean WPM readings
      if (intervalMs > 5000) {
        _resetTypingSession();
        intervalMs = null;
      }
    }

    if (_typingSessionStart == null) {
      _typingSessionStart = now;
      _keystrokeCount = 1;
    } else {
      _keystrokeCount++;
    }

    _lastKeystrokeTime = now;

    if (_isMonitoring) {
      _sensorService.recordInteraction(
        keystrokeIntervalMs: intervalMs,
        interactionFrequency: interactionFrequency,
      );
    }
    notifyListeners();
  }

  /// Inserts a new biometric snapshot into memory (FIFO queue).
  void _addDataPoint(BehaviorData data) {
    if (_collectedData.length >= maxBufferSize) {
      _collectedData.removeAt(0); // Evict oldest
    }
    _collectedData.add(data);
    
    notifyListeners();
  }

  /// Logs interaction times to track the dynamic sliding frequency window.
  void _logInteractionTimestamp() {
    _interactionTimestamps.add(DateTime.now());
    _cleanInteractionTimestamps();
  }

  /// Drops expired timestamps from the 60-second window.
  void _cleanInteractionTimestamps() {
    final cutoff = DateTime.now().subtract(_frequencyWindow);
    _interactionTimestamps.removeWhere((time) => time.isBefore(cutoff));
  }

  /// Reset typing metrics sessions.
  void _resetTypingSession() {
    _typingSessionStart = null;
    _lastKeystrokeTime = null;
    _keystrokeCount = 0;
  }

  /// Exports collected behavior biometrics as a CSV formatted String.
  String exportToCsv() {
    return BehaviorData.toCsv(_collectedData);
  }

  /// Exports collected behavior biometrics as a JSON list String.
  String exportToJson() {
    return BehaviorData.toJsonList(_collectedData);
  }

  /// Clears in-memory cache lists and resets calculation states.
  void clearData() {
    _collectedData.clear();
    _interactionTimestamps.clear();
    _resetTypingSession();
    _ownerProbability = 0.0;
    _isAuthenticated = false;
    _behaviorService.clearBuffer();
    notifyListeners();
  }

  @override
  void dispose() {
    _sensorSubscription?.cancel();
    _sensorService.dispose();
    _tfliteService.dispose();
    _behaviorService.dispose();
    super.dispose();
  }
}
