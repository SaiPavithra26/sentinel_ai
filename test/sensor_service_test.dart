import 'dart:convert';
import 'package:flutter_test/flutter_test.dart';
import 'package:flutter/services.dart';
import 'package:sentinel_ai/models/behavior_data.dart';
import 'package:sentinel_ai/providers/sensor_provider.dart';

void main() {
  // Setup Flutter test binding
  TestWidgetsFlutterBinding.ensureInitialized();

  const MethodChannel deviceChannel = MethodChannel('plugins.flutter.io/device_info');
  const MethodChannel permissionChannel = MethodChannel('flutter.baseflow.com/permissions/methods');

  setUp(() {
    // Setup Mock device info handler
    TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
        .setMockMethodCallHandler(deviceChannel, (MethodCall methodCall) async {
      if (methodCall.method == 'getAndroidDeviceInfo') {
        return {
          'brand': 'Google',
          'model': 'Pixel 6 Pro',
          'name': 'Pixel 6 Pro Mock',
        };
      }
      return null;
    });

    // Setup Mock permission request handler
    TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
        .setMockMethodCallHandler(permissionChannel, (MethodCall methodCall) async {
      if (methodCall.method == 'checkPermissionStatus') {
        return 1; // PermissionStatus.granted
      } else if (methodCall.method == 'requestPermissions') {
        return {
          12: 1, // Sensors permission index (granted)
          20: 1, // Activity recognition index (granted)
        };
      }
      return null;
    });
  });

  tearDown(() {
    TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
        .setMockMethodCallHandler(deviceChannel, null);
    TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
        .setMockMethodCallHandler(permissionChannel, null);
  });

  group('BehaviorData Model Serialization & CSV Exporters', () {
    test('toMap() and fromMap() roundtrip verification', () {
      final original = BehaviorData(
        timestamp: 1700000000000,
        accelerometerX: 0.55,
        accelerometerY: 9.81,
        accelerometerZ: -1.23,
        gyroscopeX: 0.05,
        gyroscopeY: 0.12,
        gyroscopeZ: -0.88,
        orientation: 'portrait',
        touchX: 250.0,
        touchY: 480.0,
        swipeSpeed: 3.12,
        swipeDistance: 245.5,
        swipeDirection: 'Right (12°)',
        swipeDurationMs: 80,
        tapDurationMs: null,
        interactionFrequency: 4.0,
        keystrokeIntervalMs: 180,
        deviceName: 'Google Pixel 6 Pro Mock',
      );

      final map = original.toMap();
      expect(map['timestamp'], 1700000000000);
      expect(map['accelerometerX'], 0.55);
      expect(map['gyroscopeZ'], -0.88);
      expect(map['orientation'], 'portrait');
      expect(map['touchX'], 250.0);
      expect(map['swipeDirection'], 'Right (12°)');
      expect(map['keystrokeIntervalMs'], 180);
      expect(map['deviceName'], 'Google Pixel 6 Pro Mock');

      final reconstructed = BehaviorData.fromMap(map);
      expect(reconstructed.timestamp, original.timestamp);
      expect(reconstructed.accelerometerX, original.accelerometerX);
      expect(reconstructed.accelerometerY, original.accelerometerY);
      expect(reconstructed.gyroscopeZ, original.gyroscopeZ);
      expect(reconstructed.orientation, original.orientation);
      expect(reconstructed.touchX, original.touchX);
      expect(reconstructed.swipeDirection, original.swipeDirection);
      expect(reconstructed.keystrokeIntervalMs, original.keystrokeIntervalMs);
      expect(reconstructed.deviceName, original.deviceName);
    });

    test('toCsv() correctly formats tabular behavior readings and handles comma escapes', () {
      final dataPoints = [
        BehaviorData(
          timestamp: 1700000000000,
          accelerometerX: -0.15,
          accelerometerY: 9.81,
          accelerometerZ: 0.05,
          orientation: 'portrait',
          deviceName: 'Pixel, 6 Pro Mock', // contains comma
        ),
      ];

      final csvString = BehaviorData.toCsv(dataPoints);
      
      // Ensure header and rows are present
      expect(csvString, contains('timestamp,accelerometer_x,accelerometer_y'));
      expect(csvString, contains('1700000000000,-0.15,9.81,0.05'));
      
      // Verification of CSV quotes/comma escaping
      expect(csvString, contains('"Pixel, 6 Pro Mock"'));
    });

    test('toJsonList() creates a valid JSON list representation', () {
      final dataPoints = [
        BehaviorData(timestamp: 1700000000000, accelerometerX: 1.0, orientation: 'portrait'),
        BehaviorData(timestamp: 1700000001000, accelerometerX: 2.0, orientation: 'landscape'),
      ];

      final jsonStr = BehaviorData.toJsonList(dataPoints);
      final List decoded = jsonDecode(jsonStr);

      expect(decoded.length, 2);
      expect(decoded[0]['timestamp'], 1700000000000);
      expect(decoded[0]['accelerometerX'], 1.0);
      expect(decoded[1]['timestamp'], 1700000001000);
      expect(decoded[1]['accelerometerX'], 2.0);
    });
  });

  group('SensorProvider Biometric State Calculations', () {
    test('Calculates typing Words-Per-Minute (WPM) speed estimation', () async {
      final provider = SensorProvider();
      expect(provider.typingSpeedWpm, 0.0);

      // Register first key event (starts session timer)
      provider.registerKeystroke();
      
      // Simulate keystroke ticks
      provider.registerKeystroke();
      provider.registerKeystroke();
      provider.registerKeystroke();
      provider.registerKeystroke();
      provider.registerKeystroke(); // Total 6 keys
      
      // Wait minor duration to record time elapsed
      await Future.delayed(const Duration(milliseconds: 10));

      final wpm = provider.typingSpeedWpm;
      expect(wpm, isPositive);
    });

    test('Calculates rolling window interaction frequency dynamically', () {
      final provider = SensorProvider();
      expect(provider.interactionFrequency, 0.0);

      // Perform user interactions
      provider.registerTap(x: 100, y: 150, durationMs: 65);
      provider.registerTap(x: 120, y: 140, durationMs: 70);
      provider.registerSwipe(
        startX: 50,
        startY: 50,
        distance: 200,
        direction: 'Left',
        speed: 1.5,
        durationMs: 120,
      );
      provider.registerKeystroke();

      // Total 4 interactions
      expect(provider.interactionFrequency, 4.0);
    });

    test('Data cleaning and memory buffer operations', () {
      final provider = SensorProvider();
      
      // Simulate buffer clears
      provider.clearData();
      expect(provider.dataCount, 0);
      expect(provider.collectedData.isEmpty, true);
    });
  });
}
