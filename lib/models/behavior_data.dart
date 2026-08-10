import 'dart:convert';

/// A production-ready model representing a snapshot of behavioral biometric data
/// collected from the smartphone.
class BehaviorData {
  /// Unix timestamp in milliseconds when this reading was captured.
  final int timestamp;

  // --- Physical Sensors ---
  
  /// Linear acceleration along the X axis in m/s^2 (includes gravity).
  final double? accelerometerX;
  /// Linear acceleration along the Y axis in m/s^2.
  final double? accelerometerY;
  /// Linear acceleration along the Z axis in m/s^2.
  final double? accelerometerZ;

  /// Rate of rotation around the X axis in rad/s.
  final double? gyroscopeX;
  /// Rate of rotation around the Y axis in rad/s.
  final double? gyroscopeY;
  /// Rate of rotation around the Z axis in rad/s.
  final double? gyroscopeZ;

  /// Device orientation, e.g., 'portraitUp', 'portraitDown', 'landscapeLeft', 'landscapeRight'.
  final String? orientation;

  // --- Touch and Gesture Dynamics ---

  /// The X coordinate of the screen interaction.
  final double? touchX;
  /// The Y coordinate of the screen interaction.
  final double? touchY;

  /// Speed of the swipe gesture in pixels per millisecond (px/ms).
  final double? swipeSpeed;
  /// Total distance of the swipe gesture in pixels.
  final double? swipeDistance;
  /// Direction of the swipe, e.g., 'Up', 'Down', 'Left', 'Right', or 'None'.
  final String? swipeDirection;
  /// Duration of the swipe gesture in milliseconds.
  final int? swipeDurationMs;

  /// Duration of the tap gesture (down-to-up transition) in milliseconds.
  final int? tapDurationMs;

  /// Frequency of user interactions measured as occurrences per minute.
  final double? interactionFrequency;

  // --- Keystroke Dynamics ---

  /// Time interval between consecutive key presses in milliseconds (flight time / latency).
  final int? keystrokeIntervalMs;

  // --- Metadata ---
  
  /// Hardware or device name/model identifier.
  final String? deviceName;

  const BehaviorData({
    required this.timestamp,
    this.accelerometerX,
    this.accelerometerY,
    this.accelerometerZ,
    this.gyroscopeX,
    this.gyroscopeY,
    this.gyroscopeZ,
    this.orientation,
    this.touchX,
    this.touchY,
    this.swipeSpeed,
    this.swipeDistance,
    this.swipeDirection,
    this.swipeDurationMs,
    this.tapDurationMs,
    this.interactionFrequency,
    this.keystrokeIntervalMs,
    this.deviceName,
  });

  /// Factory constructor to create a [BehaviorData] snapshot from a Map.
  factory BehaviorData.fromMap(Map<String, dynamic> map) {
    return BehaviorData(
      timestamp: map['timestamp'] as int,
      accelerometerX: (map['accelerometerX'] as num?)?.toDouble(),
      accelerometerY: (map['accelerometerY'] as num?)?.toDouble(),
      accelerometerZ: (map['accelerometerZ'] as num?)?.toDouble(),
      gyroscopeX: (map['gyroscopeX'] as num?)?.toDouble(),
      gyroscopeY: (map['gyroscopeY'] as num?)?.toDouble(),
      gyroscopeZ: (map['gyroscopeZ'] as num?)?.toDouble(),
      orientation: map['orientation'] as String?,
      touchX: (map['touchX'] as num?)?.toDouble(),
      touchY: (map['touchY'] as num?)?.toDouble(),
      swipeSpeed: (map['swipeSpeed'] as num?)?.toDouble(),
      swipeDistance: (map['swipeDistance'] as num?)?.toDouble(),
      swipeDirection: map['swipeDirection'] as String?,
      swipeDurationMs: map['swipeDurationMs'] as int?,
      tapDurationMs: map['tapDurationMs'] as int?,
      interactionFrequency: (map['interactionFrequency'] as num?)?.toDouble(),
      keystrokeIntervalMs: map['keystrokeIntervalMs'] as int?,
      deviceName: map['deviceName'] as String?,
    );
  }

  /// Converts this model instance into a JSON-compatible map.
  Map<String, dynamic> toMap() {
    return {
      'timestamp': timestamp,
      if (accelerometerX != null) 'accelerometerX': accelerometerX,
      if (accelerometerY != null) 'accelerometerY': accelerometerY,
      if (accelerometerZ != null) 'accelerometerZ': accelerometerZ,
      if (gyroscopeX != null) 'gyroscopeX': gyroscopeX,
      if (gyroscopeY != null) 'gyroscopeY': gyroscopeY,
      if (gyroscopeZ != null) 'gyroscopeZ': gyroscopeZ,
      if (orientation != null) 'orientation': orientation,
      if (touchX != null) 'touchX': touchX,
      if (touchY != null) 'touchY': touchY,
      if (swipeSpeed != null) 'swipeSpeed': swipeSpeed,
      if (swipeDistance != null) 'swipeDistance': swipeDistance,
      if (swipeDirection != null) 'swipeDirection': swipeDirection,
      if (swipeDurationMs != null) 'swipeDurationMs': swipeDurationMs,
      if (tapDurationMs != null) 'tapDurationMs': tapDurationMs,
      if (interactionFrequency != null) 'interactionFrequency': interactionFrequency,
      if (keystrokeIntervalMs != null) 'keystrokeIntervalMs': keystrokeIntervalMs,
      if (deviceName != null) 'deviceName': deviceName,
    };
  }

  /// Helper to convert a list of [BehaviorData] objects into a JSON list string.
  static String toJsonList(List<BehaviorData> dataList) {
    final list = dataList.map((e) => e.toMap()).toList();
    return jsonEncode(list);
  }

  /// Helper to convert a list of [BehaviorData] objects to a CSV string.
  /// Ideal for machine learning model training.
  static String toCsv(List<BehaviorData> dataList) {
    if (dataList.isEmpty) return '';

    // CSV Headers
    final List<String> headers = [
      'timestamp',
      'accelerometer_x',
      'accelerometer_y',
      'accelerometer_z',
      'gyroscope_x',
      'gyroscope_y',
      'gyroscope_z',
      'orientation',
      'touch_x',
      'touch_y',
      'swipe_speed',
      'swipe_distance',
      'swipe_direction',
      'swipe_duration_ms',
      'tap_duration_ms',
      'interaction_frequency',
      'keystroke_interval_ms',
      'device_name'
    ];

    final buffer = StringBuffer();
    // Write header line
    buffer.writeln(headers.join(','));

    // Write data rows
    for (final data in dataList) {
      final List<String> row = [
        data.timestamp.toString(),
        data.accelerometerX?.toString() ?? '',
        data.accelerometerY?.toString() ?? '',
        data.accelerometerZ?.toString() ?? '',
        data.gyroscopeX?.toString() ?? '',
        data.gyroscopeY?.toString() ?? '',
        data.gyroscopeZ?.toString() ?? '',
        data.escapeCsv(data.orientation),
        data.touchX?.toString() ?? '',
        data.touchY?.toString() ?? '',
        data.swipeSpeed?.toString() ?? '',
        data.swipeDistance?.toString() ?? '',
        data.escapeCsv(data.swipeDirection),
        data.swipeDurationMs?.toString() ?? '',
        data.tapDurationMs?.toString() ?? '',
        data.interactionFrequency?.toString() ?? '',
        data.keystrokeIntervalMs?.toString() ?? '',
        data.escapeCsv(data.deviceName),
      ];
      buffer.writeln(row.join(','));
    }

    return buffer.toString();
  }

  /// Escapes CSV values to handle commas, newlines, and quotes.
  String escapeCsv(String? val) {
    if (val == null) return '';
    if (val.contains(',') || val.contains('"') || val.contains('\n') || val.contains('\r')) {
      final escaped = val.replaceAll('"', '""');
      return '"$escaped"';
    }
    return val;
  }

  /// Create a copy of [BehaviorData] with updated fields.
  BehaviorData copyWith({
    int? timestamp,
    double? accelerometerX,
    double? accelerometerY,
    double? accelerometerZ,
    double? gyroscopeX,
    double? gyroscopeY,
    double? gyroscopeZ,
    String? orientation,
    double? touchX,
    double? touchY,
    double? swipeSpeed,
    double? swipeDistance,
    String? swipeDirection,
    int? swipeDurationMs,
    int? tapDurationMs,
    double? interactionFrequency,
    int? keystrokeIntervalMs,
    String? deviceName,
  }) {
    return BehaviorData(
      timestamp: timestamp ?? this.timestamp,
      accelerometerX: accelerometerX ?? this.accelerometerX,
      accelerometerY: accelerometerY ?? this.accelerometerY,
      accelerometerZ: accelerometerZ ?? this.accelerometerZ,
      gyroscopeX: gyroscopeX ?? this.gyroscopeX,
      gyroscopeY: gyroscopeY ?? this.gyroscopeY,
      gyroscopeZ: gyroscopeZ ?? this.gyroscopeZ,
      orientation: orientation ?? this.orientation,
      touchX: touchX ?? this.touchX,
      touchY: touchY ?? this.touchY,
      swipeSpeed: swipeSpeed ?? this.swipeSpeed,
      swipeDistance: swipeDistance ?? this.swipeDistance,
      swipeDirection: swipeDirection ?? this.swipeDirection,
      swipeDurationMs: swipeDurationMs ?? this.swipeDurationMs,
      tapDurationMs: tapDurationMs ?? this.tapDurationMs,
      interactionFrequency: interactionFrequency ?? this.interactionFrequency,
      keystrokeIntervalMs: keystrokeIntervalMs ?? this.keystrokeIntervalMs,
      deviceName: deviceName ?? this.deviceName,
    );
  }
}
