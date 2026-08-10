import 'dart:async';
import 'dart:convert';
import 'dart:developer' as developer;
import 'package:flutter/widgets.dart';
import 'package:sensors_plus/sensors_plus.dart';
import 'package:workmanager/workmanager.dart';
import 'package:flutter_secure_storage/flutter_secure_storage.dart';
import '../models/behavior_data.dart';
import 'sensor_service.dart';
import 'tflite_service.dart';
import 'risk_engine.dart';
import 'decision_engine.dart';
import 'feature_extractor.dart';

/// Global callback dispatcher for Workmanager background execution.
/// Runs in a separate background isolate when triggered by the OS.
@pragma('vm:entry-point')
void callbackDispatcher() {
  Workmanager().executeTask((taskName, inputData) async {
    // Ensure Flutter framework binding is initialized in background isolate
    WidgetsFlutterBinding.ensureInitialized();
    developer.log('BackgroundService: Executing task: $taskName', name: 'BackgroundService');

    try {
      if (taskName == BackgroundService.periodicBiometricCheckTask) {
        final secureStorage = const FlutterSecureStorage();

        // 1. Load active Owner ID saved by the UI (defaults to Owner 1)
        final ownerId = await secureStorage.read(key: 'active_owner_id') ?? '1';
        developer.log('BackgroundService: Evaluating risk profile for user: $ownerId', name: 'BackgroundService');

        // 2. Instantiate and load Tflite model service
        final tfliteService = TfliteInferenceService();
        await tfliteService.initialize(ownerId: ownerId);

        if (!tfliteService.isLoaded) {
          developer.log('BackgroundService: Error: Failed to load TFLite models in background. Aborting.', name: 'BackgroundService');
          return true; // Return true so OS doesn't keep retrying failed configurations
        }

        // 3. Retrieve sensor updates (sample briefly for 5 seconds to conserve battery)
        final sensorService = SensorService();
        final List<BehaviorData> sampledData = [];
        final completer = Completer<void>();

        sensorService.startMonitoring(
          sensorInterval: SensorInterval.normalInterval,
        );

        final subscription = sensorService.dataStream.listen((data) {
          sampledData.add(data);
        });

        Timer(const Duration(seconds: 5), () {
          subscription.cancel();
          sensorService.stopMonitoring();
          completer.complete();
        });

        await completer.future;

        if (sampledData.isEmpty) {
          developer.log('BackgroundService: No motion sensor data captured during sample window.', name: 'BackgroundService');
          return true;
        }

        // 4. Feature Extraction on the sampled window
        final rawFeatures = FeatureExtractor.extractFeatures(
          sampledData,
          sessionDurationMins: 5.0,
        );

        // 5. Run TFLite inference (returns owner verification confidence)
        final confidence = tfliteService.predict(rawFeatures);

        // 6. Compute Unified Risk (with default physical anomaly baselines in background)
        final riskEngine = RiskEngine();
        final riskReport = riskEngine.calculateRisk(
          tfOwnerConfidence: confidence,
          movementAnomaly: 0.0, // baseline normal
          sensorAnomaly: 0.0,
        );

        // 7. Evaluate mitigation triggers
        final decisionEngine = DecisionEngine();
        final decision = decisionEngine.evaluateRisk(riskReport.riskScore);

        // 8. Persist results to secure storage for UI response
        final reportMap = {
          'riskScore': riskReport.riskScore,
          'riskLevel': riskReport.riskLevel.index,
          'triggerLock': decision.triggerLock,
          'triggerAlert': decision.triggerAlert,
          'triggerCamera': decision.triggerCamera,
          'triggerLiveTracking': decision.triggerLiveTracking,
          'recommendation': decision.recommendationMessage,
          'timestamp': DateTime.now().toIso8601String(),
        };

        await secureStorage.write(
          key: 'last_background_security_report',
          value: jsonEncode(reportMap),
        );

        if (decision.hasTriggers) {
          developer.log('BackgroundService: THREAT DETECTED in background! Recommended Action: ${decision.recommendationMessage}', name: 'BackgroundService');
          await secureStorage.write(key: 'active_threat_detected', value: 'true');
        } else {
          await secureStorage.write(key: 'active_threat_detected', value: 'false');
        }
      }
      return true;
    } catch (e, stackTrace) {
      developer.log(
        'BackgroundService: Exception occurred inside execution callback: $e',
        error: e,
        stackTrace: stackTrace,
        name: 'BackgroundService',
      );
      return false; // Tells OS to reschedule or log failure
    }
  });
}

/// A service wrapping the Workmanager plugin to schedule periodic security risk checks.
/// 
/// Tasks are scheduled at the OS level, meaning they continue running even if
/// the application is killed/terminated or after the device is rebooted.
class BackgroundService {
  /// Unique task identifier for the periodic biometric execution.
  static const String periodicBiometricCheckTask = 'com.sentinel_ai.periodicBiometricCheckTask';

  /// Unique work registration name.
  static const String uniquePeriodicName = 'periodic_biometric_security_check';

  /// Initializes the Workmanager engine and registers the global dispatcher.
  static Future<void> initialize() async {
    try {
      developer.log('BackgroundService: Initializing background service...', name: 'BackgroundService');
      await Workmanager().initialize(
        callbackDispatcher,
        isInDebugMode: false,
      );
      developer.log('BackgroundService: Background service initialization complete.', name: 'BackgroundService');
    } catch (e) {
      developer.log('BackgroundService: Initialization failed: $e', name: 'BackgroundService');
    }
  }

  /// Schedules the periodic background security checker.
  /// 
  /// This task executes every 15 minutes (the minimum interval allowed by the Android OS).
  /// [ExistingWorkPolicy.keep] ensures that the scheduled periodic check persists across reboots
  /// and is not duplicated or overwritten on app re-launches.
  static Future<void> registerPeriodicTask() async {
    try {
      developer.log('BackgroundService: Registering periodic task...', name: 'BackgroundService');
      await Workmanager().registerPeriodicTask(
        uniquePeriodicName,
        periodicBiometricCheckTask,
        frequency: const Duration(minutes: 15),
        existingWorkPolicy: ExistingWorkPolicy.keep,
        backoffPolicy: BackoffPolicy.exponential,
        backoffPolicyDelay: const Duration(seconds: 10),
      );
      developer.log('BackgroundService: Periodic task registered successfully.', name: 'BackgroundService');
    } catch (e) {
      developer.log('BackgroundService: Task registration failed: $e', name: 'BackgroundService');
    }
  }

  /// Cancels all background tasks scheduled under this application.
  static Future<void> cancelAllTasks() async {
    try {
      await Workmanager().cancelAll();
      developer.log('BackgroundService: Cancelled all background tasks.', name: 'BackgroundService');
    } catch (e) {
      developer.log('BackgroundService: Failed to cancel tasks: $e', name: 'BackgroundService');
    }
  }
}
