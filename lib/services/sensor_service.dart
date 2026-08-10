import 'dart:async';
import 'dart:io';
import 'package:flutter/widgets.dart';
import 'package:sensors_plus/sensors_plus.dart';
import 'package:device_info_plus/device_info_plus.dart';
import '../models/behavior_data.dart';

/// A production-ready service responsible for physical sensor monitoring,
/// combining accelerometer and gyroscope updates, managing lifecycles for battery saving,
/// and injecting device orientation and name metadata.
class SensorService with WidgetsBindingObserver {
  // Stream controller to broadcast unified BehaviorData events
  final StreamController<BehaviorData> _dataStreamController =
      StreamController<BehaviorData>.broadcast();

  /// Unified stream of behavior readings.
  Stream<BehaviorData> get dataStream => _dataStreamController.stream;

  // Subscriptions to sensors_plus streams
  StreamSubscription<AccelerometerEvent>? _accelerometerSubscription;
  StreamSubscription<GyroscopeEvent>? _gyroscopeSubscription;

  // Cached sensor readings to present combined data snapshots
  double? _lastAccX;
  double? _lastAccY;
  double? _lastAccZ;

  double? _lastGyroX;
  double? _lastGyroY;
  double? _lastGyroZ;

  // Metadata properties
  String _deviceName = 'Unknown';
  final DeviceInfoPlugin _deviceInfo = DeviceInfoPlugin();

  // Configuration and state flags
  bool _isMonitoring = false;
  Duration _sensorInterval = SensorInterval.normalInterval;
  
  // Steady periodic sampling parameters (to save battery/limit output rate)
  Timer? _samplingTimer;
  Duration? _samplingPeriod;

  /// Returns true if sensor monitoring is active.
  bool get isMonitoring => _isMonitoring;

  SensorService() {
    _initDeviceInfo();
  }

  /// Retrieve device information asynchronously on initialization.
  Future<void> _initDeviceInfo() async {
    try {
      if (Platform.isAndroid) {
        final androidInfo = await _deviceInfo.androidInfo;
        _deviceName = '${androidInfo.brand} ${androidInfo.model}';
      } else if (Platform.isIOS) {
        final iosInfo = await _deviceInfo.iosInfo;
        _deviceName = iosInfo.name;
      }
    } catch (_) {
      _deviceName = Platform.operatingSystem;
    }
  }

  /// Starts behavioral sensor monitoring.
  /// 
  /// [sensorInterval] specifies how frequently the hardware sensors poll.
  /// [samplingPeriod] optionally defines a steady clock interval for emitting snapshots.
  /// If [samplingPeriod] is null, data points are emitted on every hardware interrupt.
  void startMonitoring({
    Duration sensorInterval = SensorInterval.normalInterval,
    Duration? samplingPeriod,
  }) {
    if (_isMonitoring) return;
    _isMonitoring = true;
    _sensorInterval = sensorInterval;
    _samplingPeriod = samplingPeriod;

    // Register lifecycle observer to listen to app pause/resume events
    WidgetsBinding.instance.addObserver(this);

    _startSubscriptions();
  }

  /// Stops behavioral sensor monitoring and cleans up active subscriptions.
  void stopMonitoring() {
    if (!_isMonitoring) return;
    _isMonitoring = false;

    WidgetsBinding.instance.removeObserver(this);
    _stopSubscriptions();
  }

  /// Initialize and start hardware sensor streams.
  void _startSubscriptions() {
    _stopSubscriptions(); // Reset existing subscriptions if any

    // Subscribe to Accelerometer events
    _accelerometerSubscription = accelerometerEventStream(samplingPeriod: _sensorInterval).listen(
      (AccelerometerEvent event) {
        _lastAccX = event.x;
        _lastAccY = event.y;
        _lastAccZ = event.z;
        if (_samplingPeriod == null) {
          _emitSnapshot();
        }
      },
      onError: (error) {
        debugPrint('SensorService: Accelerometer stream error: $error');
      },
    );

    // Subscribe to Gyroscope events
    _gyroscopeSubscription = gyroscopeEventStream(samplingPeriod: _sensorInterval).listen(
      (GyroscopeEvent event) {
        _lastGyroX = event.x;
        _lastGyroY = event.y;
        _lastGyroZ = event.z;
        if (_samplingPeriod == null) {
          _emitSnapshot();
        }
      },
      onError: (error) {
        debugPrint('SensorService: Gyroscope stream error: $error');
      },
    );

    // Setup periodic sampling if configured (helps throttle data rates & save power)
    if (_samplingPeriod != null) {
      _samplingTimer = Timer.periodic(_samplingPeriod!, (_) {
        _emitSnapshot();
      });
    }
  }

  /// Cancel all subscriptions and clean up timer.
  void _stopSubscriptions() {
    _accelerometerSubscription?.cancel();
    _accelerometerSubscription = null;

    _gyroscopeSubscription?.cancel();
    _gyroscopeSubscription = null;

    _samplingTimer?.cancel();
    _samplingTimer = null;
  }

  /// Combines all active state parameters into a single [BehaviorData] model and streams it.
  void _emitSnapshot({
    double? touchX,
    double? touchY,
    double? swipeSpeed,
    double? swipeDistance,
    String? swipeDirection,
    int? swipeDurationMs,
    int? tapDurationMs,
    double? interactionFrequency,
    int? keystrokeIntervalMs,
  }) {
    if (!_isMonitoring) return;

    final data = BehaviorData(
      timestamp: DateTime.now().millisecondsSinceEpoch,
      accelerometerX: _lastAccX,
      accelerometerY: _lastAccY,
      accelerometerZ: _lastAccZ,
      gyroscopeX: _lastGyroX,
      gyroscopeY: _lastGyroY,
      gyroscopeZ: _lastGyroZ,
      orientation: _getDeviceOrientation(),
      touchX: touchX,
      touchY: touchY,
      swipeSpeed: swipeSpeed,
      swipeDistance: swipeDistance,
      swipeDirection: swipeDirection,
      swipeDurationMs: swipeDurationMs,
      tapDurationMs: tapDurationMs,
      interactionFrequency: interactionFrequency,
      keystrokeIntervalMs: keystrokeIntervalMs,
      deviceName: _deviceName,
    );

    _dataStreamController.add(data);
  }

  /// Pushes an interaction event to be merged with current sensor cache and emitted instantly.
  void recordInteraction({
    double? touchX,
    double? touchY,
    double? swipeSpeed,
    double? swipeDistance,
    String? swipeDirection,
    int? swipeDurationMs,
    int? tapDurationMs,
    double? interactionFrequency,
    int? keystrokeIntervalMs,
  }) {
    _emitSnapshot(
      touchX: touchX,
      touchY: touchY,
      swipeSpeed: swipeSpeed,
      swipeDistance: swipeDistance,
      swipeDirection: swipeDirection,
      swipeDurationMs: swipeDurationMs,
      tapDurationMs: tapDurationMs,
      interactionFrequency: interactionFrequency,
      keystrokeIntervalMs: keystrokeIntervalMs,
    );
  }

  /// Resolves the current device orientation dynamically.
  String _getDeviceOrientation() {
    final dispatcher = WidgetsBinding.instance.platformDispatcher;
    if (dispatcher.views.isEmpty) return 'portrait'; // Fallback
    final view = dispatcher.views.first;
    final size = view.physicalSize;
    return size.width > size.height ? 'landscape' : 'portrait';
  }

  // --- AppLifecycleState Hooks for Battery Optimization ---

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    if (!_isMonitoring) return;

    switch (state) {
      case AppLifecycleState.paused:
      case AppLifecycleState.inactive:
        // Automatically release sensor holds when backgrounded
        _stopSubscriptions();
        debugPrint('SensorService: Backgrounded. Suspended hardware sensor streams.');
        break;
      case AppLifecycleState.resumed:
        // Restore stream holds once returned to foreground
        _startSubscriptions();
        debugPrint('SensorService: Foregrounded. Restored hardware sensor streams.');
        break;
      case AppLifecycleState.detached:
      case AppLifecycleState.hidden:
        _stopSubscriptions();
        break;
    }
  }

  /// Clean up and dispose of controller.
  void dispose() {
    stopMonitoring();
    _dataStreamController.close();
  }
}
