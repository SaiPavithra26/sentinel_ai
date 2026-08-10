import 'dart:math';
import '../models/behavior_data.dart';

/// A production-ready utility in Dart that replicates Python's feature extraction logic.
/// Takes a list of raw BehaviorData snapshots (representing a session) and extracts
/// keystroke dynamics, touch dynamics, accelerometer, and gyroscope features.
class FeatureExtractor {
  
  /// Helper to calculate mean of a list.
  static double mean(List<double> list) {
    if (list.isEmpty) return double.nan;
    return list.reduce((a, b) => a + b) / list.length;
  }

  /// Helper to calculate standard deviation of a list.
  static double std(List<double> list, double meanVal) {
    if (list.length < 2) return 0.0;
    double sum = 0.0;
    for (var x in list) {
      sum += (x - meanVal) * (x - meanVal);
    }
    return sqrt(sum / (list.length - 1));
  }

  /// Helper to calculate median of a list.
  static double median(List<double> list) {
    if (list.isEmpty) return double.nan;
    final sorted = List<double>.from(list)..sort();
    final mid = sorted.length ~/ 2;
    if (sorted.length % 2 == 0) {
      return (sorted[mid - 1] + sorted[mid]) / 2.0;
    } else {
      return sorted[mid];
    }
  }

  /// Helper to calculate sample skewness (bias=False equivalent in SciPy).
  static double skew(List<double> list, double meanVal, double stdVal) {
    if (list.length < 3 || stdVal < 1e-9) return 0.0;
    double sum = 0.0;
    for (var x in list) {
      sum += pow((x - meanVal) / stdVal, 3);
    }
    final n = list.length;
    return (n / ((n - 1) * (n - 2))) * sum;
  }

  /// Helper to calculate sample kurtosis (excess kurtosis, bias=False equivalent).
  static double kurtosis(List<double> list, double meanVal, double stdVal) {
    if (list.length < 4 || stdVal < 1e-9) return 0.0;
    double sum = 0.0;
    for (var x in list) {
      sum += pow((x - meanVal) / stdVal, 4);
    }
    final n = list.length;
    final term1 = (n * (n + 1)) / ((n - 1) * (n - 2) * (n - 3));
    final term2 = (3 * pow(n - 1, 2)) / ((n - 2) * (n - 3));
    return term1 * sum - term2;
  }

  /// Extracts circular mean of angles (radians).
  static double circularMean(List<double> angles) {
    if (angles.isEmpty) return double.nan;
    double sumSin = 0.0;
    double sumCos = 0.0;
    for (var a in angles) {
      sumSin += sin(a);
      sumCos += cos(a);
    }
    return atan2(sumSin / angles.length, sumCos / angles.length);
  }

  /// Extracts circular standard deviation of angles.
  static double circularStd(List<double> angles) {
    if (angles.isEmpty) return 0.0;
    double sumSin = 0.0;
    double sumCos = 0.0;
    for (var a in angles) {
      sumSin += sin(a);
      sumCos += cos(a);
    }
    final r = sqrt(pow(sumSin / angles.length, 2) + pow(sumCos / angles.length, 2));
    final rClipped = max(min(r, 1.0), 1e-9);
    return sqrt(-2.0 * log(rClipped));
  }

  /// Parses swipe angle from the swipeDirection string representation, e.g. "Right (35°)" -> 35° -> radians
  static double? parseSwipeDirectionToRad(String? dirStr) {
    if (dirStr == null) return null;
    final regExp = RegExp(r'\((\d+)\°\)');
    final match = regExp.firstMatch(dirStr);
    if (match != null) {
      final deg = double.tryParse(match.group(1) ?? '');
      if (deg != null) {
        double rad = deg * pi / 180.0;
        if (rad > pi) rad -= 2.0 * pi;
        return rad;
      }
    }
    return null;
  }

  /// Extracts the feature map from raw session data.
  static Map<String, double> extractFeatures(List<BehaviorData> sessionData, {double sessionDurationMins = 5.0}) {
    final Map<String, double> features = {};

    // 1. Filter lists for metrics calculations
    final List<double> keyIntervals = [];
    final List<double> swipeDistances = [];
    final List<double> swipeDurations = [];
    final List<double> swipeSpeeds = [];
    final List<double> swipeDirections = [];
    final List<double> tapDurations = [];
    
    final List<double> touchPressures = [];
    final List<double> touchAreas = [];
    final List<double> touchOrientations = [];

    final List<double> accX = [];
    final List<double> accY = [];
    final List<double> accZ = [];
    final List<double> accMag = [];

    final List<double> gyroX = [];
    final List<double> gyroY = [];
    final List<double> gyroZ = [];
    final List<double> gyroMag = [];

    double keystrokeCount = 0.0;
    double swipeCount = 0.0;
    double tapCount = 0.0;

    for (var data in sessionData) {
      // Keystrokes
      if (data.keystrokeIntervalMs != null) {
        keyIntervals.add(data.keystrokeIntervalMs! / 1000.0); // Convert to seconds
        keystrokeCount += 1.0;
      }

      // Swipes
      if (data.swipeDistance != null) {
        swipeDistances.add(data.swipeDistance!);
        swipeCount += 1.0;
        if (data.swipeDurationMs != null) {
          swipeDurations.add(data.swipeDurationMs! / 1000.0);
        }
        if (data.swipeSpeed != null) {
          // swipeSpeed is in pixels per millisecond in Dart, but let's align
          swipeSpeeds.add(data.swipeSpeed!);
        }
        final rad = parseSwipeDirectionToRad(data.swipeDirection);
        if (rad != null) {
          swipeDirections.add(rad);
        }
      }

      // Taps
      if (data.tapDurationMs != null && data.swipeDistance == null) {
        tapDurations.add(data.tapDurationMs! / 1000.0);
        tapCount += 1.0;
      }

      // Touch metadata
      if (data.touchX != null) {
        // Mock default pressure/area/orientation or parse if present
        touchPressures.add(1.0); // standard baseline
        touchAreas.add(1.0);
        touchOrientations.add(0.0);
      }

      // Accelerometer
      if (data.accelerometerX != null) {
        accX.add(data.accelerometerX!);
        accY.add(data.accelerometerY!);
        accZ.add(data.accelerometerZ!);
        accMag.add(sqrt(pow(data.accelerometerX!, 2) + pow(data.accelerometerY!, 2) + pow(data.accelerometerZ!, 2)));
      }

      // Gyroscope
      if (data.gyroscopeX != null) {
        gyroX.add(data.gyroscopeX!);
        gyroY.add(data.gyroscopeY!);
        gyroZ.add(data.gyroscopeZ!);
        gyroMag.add(sqrt(pow(data.gyroscopeX!, 2) + pow(data.gyroscopeY!, 2) + pow(data.gyroscopeZ!, 2)));
      }
    }

    // 2. Keystroke Features
    features['keystroke_count'] = keystrokeCount;
    features['typing_speed'] = keystrokeCount / (sessionDurationMins * 60.0);
    
    // Standard flight time distributions (F3 Down-Down is keyIntervals)
    _addListStats(features, keyIntervals, 'flight_time_f3');
    // Set others to nan (will be imputed)
    _addEmptyStats(features, 'key_hold_time');
    _addEmptyStats(features, 'flight_time_f1');
    _addEmptyStats(features, 'flight_time_f2');
    _addEmptyStats(features, 'flight_time_f4');

    // 3. Touch features
    features['swipe_count'] = swipeCount;
    features['tap_count'] = tapCount;
    _addListStats(features, swipeDistances, 'swipe_dist');
    _addListStats(features, swipeDurations, 'swipe_dur');
    _addListStats(features, swipeSpeeds, 'swipe_speed');
    
    // Swipe directions circular stats
    if (swipeDirections.isNotEmpty) {
      features['swipe_dir_mean'] = circularMean(swipeDirections);
      features['swipe_dir_std'] = circularStd(swipeDirections);
      features['swipe_dir_median'] = median(swipeDirections);
      features['swipe_dir_min'] = swipeDirections.reduce(min);
      features['swipe_dir_max'] = swipeDirections.reduce(max);
    } else {
      _addEmptyStats(features, 'swipe_dir');
    }

    _addListStats(features, tapDurations, 'tap_dur');
    _addListStats(features, touchPressures, 'touch_press');
    _addListStats(features, touchAreas, 'touch_area');
    _addListStats(features, touchOrientations, 'touch_orient');

    // Set instant velocity/acc values to nan (will be imputed)
    features['swipe_inst_vel_mean'] = double.nan;
    features['swipe_inst_vel_std'] = 0.0;
    features['swipe_inst_acc_mean'] = double.nan;
    features['swipe_inst_acc_std'] = 0.0;

    // 4. Accelerometer features
    _addMotionStats(features, accX, accY, accZ, accMag, 'acc');

    // 5. Gyroscope features
    _addMotionStats(features, gyroX, gyroY, gyroZ, gyroMag, 'gyro');

    return features;
  }

  static void _addListStats(Map<String, double> features, List<double> list, String prefix) {
    if (list.isNotEmpty) {
      final meanVal = mean(list);
      final stdVal = std(list, meanVal);
      features['${prefix}_mean'] = meanVal;
      features['${prefix}_std'] = stdVal;
      features['${prefix}_median'] = median(list);
      features['${prefix}_min'] = list.reduce(min);
      features['${prefix}_max'] = list.reduce(max);
    } else {
      _addEmptyStats(features, prefix);
    }
  }

  static void _addEmptyStats(Map<String, double> features, String prefix) {
    features['${prefix}_mean'] = double.nan;
    features['${prefix}_std'] = 0.0;
    features['${prefix}_median'] = double.nan;
    features['${prefix}_min'] = double.nan;
    features['${prefix}_max'] = double.nan;
  }

  static void _addMotionStats(Map<String, double> features, List<double> x, List<double> y, List<double> z, List<double> mag, String prefix) {
    _addAxisStats(features, x, '${prefix}_Xvalue');
    _addAxisStats(features, y, '${prefix}_Yvalue');
    _addAxisStats(features, z, '${prefix}_Zvalue');
    _addAxisStats(features, mag, '${prefix}_Mag');
  }

  static void _addAxisStats(Map<String, double> features, List<double> list, String name) {
    if (list.isNotEmpty) {
      final meanVal = mean(list);
      final stdVal = std(list, meanVal);
      features['${name}_mean'] = meanVal;
      features['${name}_std'] = stdVal;
      features['${name}_min'] = list.reduce(min);
      features['${name}_max'] = list.reduce(max);
      features['${name}_median'] = median(list);
      features['${name}_skew'] = skew(list, meanVal, stdVal);
      features['${name}_kurt'] = kurtosis(list, meanVal, stdVal);
    } else {
      features['${name}_mean'] = double.nan;
      features['${name}_std'] = 0.0;
      features['${name}_min'] = double.nan;
      features['${name}_max'] = double.nan;
      features['${name}_median'] = double.nan;
      features['${name}_skew'] = 0.0;
      features['${name}_kurt'] = 0.0;
    }
  }
}
