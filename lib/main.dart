import 'dart:math';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:provider/provider.dart';
import 'package:sensors_plus/sensors_plus.dart';

import 'models/behavior_data.dart';
import 'providers/sensor_provider.dart';
import 'services/background_service.dart';

void main() async {
  // Ensure widget bindings are initialized before permission checks or platform views
  WidgetsFlutterBinding.ensureInitialized();
  
  // Initialize and register background service tasks
  await BackgroundService.initialize();
  await BackgroundService.registerPeriodicTask();

  // Set preferred orientations
  SystemChrome.setPreferredOrientations([
    DeviceOrientation.portraitUp,
    DeviceOrientation.portraitDown,
  ]);

  runApp(
    ChangeNotifierProvider(
      create: (_) => SensorProvider(),
      child: const SentinelApp(),
    ),
  );
}

class SentinelApp extends StatelessWidget {
  const SentinelApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'SentinelAI Behavioral Biometrics',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        brightness: Brightness.dark,
        scaffoldBackgroundColor: const Color(0xFF0F0F16),
        primaryColor: const Color(0xFF6366F1),
        colorScheme: const ColorScheme.dark(
          primary: Color(0xFF6366F1),
          secondary: Color(0xFF10B981),
          surface: Color(0xFF181824),
          error: Color(0xFFEF4444),
        ),
        fontFamily: 'Roboto',
      ),
      home: const DashboardScreen(),
    );
  }
}

class DashboardScreen extends StatefulWidget {
  const DashboardScreen({super.key});

  @override
  State<DashboardScreen> createState() => _DashboardScreenState();
}

class _DashboardScreenState extends State<DashboardScreen> {
  // Collection settings state
  Duration _selectedInterval = SensorInterval.normalInterval;
  Duration? _selectedSamplingPeriod; // null = continuous stream
  
  final Map<String, Duration> _intervals = {
    'Normal (200ms)': SensorInterval.normalInterval,
    'UI (60ms)': SensorInterval.uiInterval,
    'Game (20ms)': SensorInterval.gameInterval,
  };

  final Map<String, Duration?> _samplingPeriods = {
    'Continuous (Raw)': null,
    'Sample 100ms': const Duration(milliseconds: 100),
    'Sample 250ms': const Duration(milliseconds: 250),
    'Sample 500ms': const Duration(milliseconds: 500),
  };

  @override
  Widget build(BuildContext context) {
    final provider = Provider.of<SensorProvider>(context);
    final lastData = provider.collectedData.isNotEmpty ? provider.collectedData.last : null;

    return Scaffold(
      appBar: AppBar(
        title: const Text(
          'SentinelAI',
          style: TextStyle(fontWeight: FontWeight.w900, letterSpacing: 1.2, color: Colors.white),
        ),
        centerTitle: true,
        backgroundColor: const Color(0xFF12121E),
        elevation: 0,
        actions: [
          IconButton(
            icon: const Icon(Icons.info_outline, color: Colors.indigoAccent),
            onPressed: () => _showExplainDialog(context),
          ),
        ],
      ),
      body: SingleChildScrollView(
        physics: const BouncingScrollPhysics(),
        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            // Header collection status card
            _buildStatusHeader(provider),
            const SizedBox(height: 16),

            // Hardware configuration controls
            _buildConfigCard(provider),
            const SizedBox(height: 16),

            // Live Sensor stream metrics
            _buildSensorsCard(lastData),
            const SizedBox(height: 16),

            // Interactive gesture pad
            _buildInteractionCard(provider),
            const SizedBox(height: 16),

            // Typing metrics capturer
            _buildTypingCard(provider),
            const SizedBox(height: 16),

            // In-Memory log and Export control panel
            _buildLogControlCard(context, provider),
            const SizedBox(height: 24),
          ],
        ),
      ),
    );
  }

  Widget _buildStatusHeader(SensorProvider provider) {
    Color statusColor = const Color(0xFFEF4444);
    String statusText = 'MONITORING INACTIVE';
    String description = 'Sensor lines are idle. Tap below to begin tracking';
    
    if (provider.isMonitoring) {
      if (provider.isModelLoaded) {
        statusColor = provider.isAuthenticated ? const Color(0xFF10B981) : const Color(0xFFEF4444);
        statusText = provider.isAuthenticated ? 'AUTHENTICATED (OWNER)' : 'UNAUTHORIZED (SUSPICIOUS)';
        description = 'Owner Probability: ${(provider.ownerProbability * 100.0).toStringAsFixed(1)}% (Threshold: ${(provider.threshold * 100.0).toStringAsFixed(1)}%)';
      } else {
        statusColor = const Color(0xFF6366F1);
        statusText = 'MONITORING ACTIVE';
        description = 'Biometric features streaming. Model loading...';
      }
    }

    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: const Color(0xFF181824),
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: statusColor.withOpacity(0.3), width: 1.5),
        boxShadow: [
          BoxShadow(
            color: statusColor.withOpacity(0.08),
            blurRadius: 16,
            offset: const Offset(0, 4),
          )
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          Row(
            children: [
              Container(
                width: 12,
                height: 12,
                decoration: BoxDecoration(
                  color: statusColor,
                  shape: BoxShape.circle,
                  boxShadow: [
                    BoxShadow(
                      color: statusColor.withOpacity(0.5),
                      blurRadius: 8,
                      spreadRadius: 2,
                    )
                  ],
                ),
              ),
              const SizedBox(width: 12),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      statusText,
                      style: TextStyle(
                        fontWeight: FontWeight.bold,
                        fontSize: 14,
                        color: statusColor,
                        letterSpacing: 0.5,
                      ),
                    ),
                    const SizedBox(height: 2),
                    Text(
                      description,
                      style: TextStyle(
                        color: Colors.white.withOpacity(0.7),
                        fontSize: 12,
                      ),
                    ),
                  ],
                ),
              ),
              ElevatedButton(
                onPressed: () async {
                  if (provider.isMonitoring) {
                    await provider.stopCollection();
                  } else {
                    final success = await provider.startCollection(
                      sensorInterval: _selectedInterval,
                      samplingPeriod: _selectedSamplingPeriod,
                      ownerId: '1', // default owner ID for testing
                    );
                    if (!success && mounted) {
                      ScaffoldMessenger.of(context).showSnackBar(
                        const SnackBar(
                          content: Text('Permissions required to begin behavioral tracking.'),
                          backgroundColor: Colors.redAccent,
                        ),
                      );
                    }
                  }
                },
                style: ElevatedButton.styleFrom(
                  backgroundColor: provider.isMonitoring ? const Color(0xFFEF4444) : const Color(0xFF6366F1),
                  foregroundColor: Colors.white,
                  elevation: 4,
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(12),
                  ),
                  padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
                ),
                child: Text(
                  provider.isMonitoring ? 'Stop' : 'Start',
                  style: const TextStyle(fontWeight: FontWeight.bold, fontSize: 14),
                ),
              ),
            ],
          ),
          if (provider.isMonitoring && provider.isModelLoaded) ...[
            const SizedBox(height: 12),
            ClipRRect(
              borderRadius: BorderRadius.circular(4),
              child: LinearProgressIndicator(
                value: provider.ownerProbability.clamp(0.0, 1.0),
                backgroundColor: Colors.white10,
                valueColor: AlwaysStoppedAnimation<Color>(statusColor),
                minHeight: 6,
              ),
            ),
            const SizedBox(height: 4),
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                Text(
                  'Suspicious User',
                  style: TextStyle(fontSize: 10, color: Colors.white.withOpacity(0.4)),
                ),
                Text(
                  'Owner',
                  style: TextStyle(fontSize: 10, color: Colors.white.withOpacity(0.4)),
                ),
              ],
            )
          ]
        ],
      ),
    );
  }

  Widget _buildConfigCard(SensorProvider provider) {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: const Color(0xFF181824),
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: Colors.white.withOpacity(0.06)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          Row(
            children: [
              const Icon(Icons.settings_outlined, color: Color(0xFF6366F1)),
              const SizedBox(width: 8),
              const Text(
                'Sensor Configurations & Battery Saver',
                style: TextStyle(fontWeight: FontWeight.bold, fontSize: 15, color: Colors.white),
              ),
            ],
          ),
          const SizedBox(height: 12),
          Row(
            children: [
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text('Hardware Interval', style: TextStyle(color: Colors.white.withOpacity(0.5), fontSize: 11)),
                    const SizedBox(height: 4),
                    DropdownButtonFormField<Duration>(
                      value: _selectedInterval,
                      dropdownColor: const Color(0xFF1E1E2D),
                      decoration: const InputDecoration(
                        isDense: true,
                        contentPadding: EdgeInsets.symmetric(horizontal: 10, vertical: 8),
                        border: OutlineInputBorder(),
                      ),
                      style: const TextStyle(color: Colors.white, fontSize: 13),
                      items: _intervals.entries.map((e) {
                        return DropdownMenuItem<Duration>(
                          value: e.value,
                          child: Text(e.key),
                        );
                      }).toList(),
                      onChanged: provider.isMonitoring
                          ? null
                          : (val) {
                              if (val != null) setState(() => _selectedInterval = val);
                            },
                    ),
                  ],
                ),
              ),
              const SizedBox(width: 12),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text('Battery Sample Rate', style: TextStyle(color: Colors.white.withOpacity(0.5), fontSize: 11)),
                    const SizedBox(height: 4),
                    DropdownButtonFormField<Duration?>(
                      value: _selectedSamplingPeriod,
                      dropdownColor: const Color(0xFF1E1E2D),
                      decoration: const InputDecoration(
                        isDense: true,
                        contentPadding: EdgeInsets.symmetric(horizontal: 10, vertical: 8),
                        border: OutlineInputBorder(),
                      ),
                      style: const TextStyle(color: Colors.white, fontSize: 13),
                      items: _samplingPeriods.entries.map((e) {
                        return DropdownMenuItem<Duration?>(
                          value: e.value,
                          child: Text(e.key),
                        );
                      }).toList(),
                      onChanged: provider.isMonitoring
                          ? null
                          : (val) {
                              setState(() => _selectedSamplingPeriod = val);
                            },
                    ),
                  ],
                ),
              ),
            ],
          ),
          if (provider.isMonitoring) ...[
            const SizedBox(height: 8),
            Text(
              '⚠️ Stop monitoring to modify hardware configurations.',
              style: TextStyle(color: Colors.amberAccent.withOpacity(0.8), fontSize: 11),
              textAlign: TextAlign.center,
            ),
          ]
        ],
      ),
    );
  }

  Widget _buildSensorsCard(BehaviorData? data) {
    final hasData = data != null;

    final accX = hasData ? (data.accelerometerX?.toStringAsFixed(3) ?? '0.000') : '0.000';
    final accY = hasData ? (data.accelerometerY?.toStringAsFixed(3) ?? '0.000') : '0.000';
    final accZ = hasData ? (data.accelerometerZ?.toStringAsFixed(3) ?? '0.000') : '0.000';

    final gyroX = hasData ? (data.gyroscopeX?.toStringAsFixed(3) ?? '0.000') : '0.000';
    final gyroY = hasData ? (data.gyroscopeY?.toStringAsFixed(3) ?? '0.000') : '0.000';
    final gyroZ = hasData ? (data.gyroscopeZ?.toStringAsFixed(3) ?? '0.000') : '0.000';

    final orientation = hasData ? (data.orientation ?? 'Portrait') : 'Portrait';
    final device = hasData ? (data.deviceName ?? 'Device') : 'Device (Loading)';

    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: const Color(0xFF181824),
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: Colors.white.withOpacity(0.06)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Row(
                children: [
                  const Icon(Icons.sensors, color: Color(0xFF6366F1)),
                  const SizedBox(width: 8),
                  const Text(
                    'Physical Sensors (Live Cache)',
                    style: TextStyle(fontWeight: FontWeight.bold, fontSize: 15, color: Colors.white),
                  ),
                ],
              ),
              Container(
                padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                decoration: BoxDecoration(
                  color: Colors.white.withOpacity(0.05),
                  borderRadius: BorderRadius.circular(8),
                ),
                child: Text(
                  orientation.toUpperCase(),
                  style: const TextStyle(fontSize: 10, fontWeight: FontWeight.bold, color: Colors.indigoAccent),
                ),
              ),
            ],
          ),
          const SizedBox(height: 12),
          Text(
            'Hardware Source: $device',
            style: TextStyle(color: Colors.white.withOpacity(0.4), fontSize: 11, fontStyle: FontStyle.italic),
          ),
          const SizedBox(height: 12),
          Row(
            children: [
              Expanded(
                child: _buildSensorDisplay(
                  'ACCELEROMETER (m/s²)',
                  ['X: $accX', 'Y: $accY', 'Z: $accZ'],
                  const Color(0xFF6366F1),
                  [data?.accelerometerX, data?.accelerometerY, data?.accelerometerZ],
                ),
              ),
              const SizedBox(width: 12),
              Expanded(
                child: _buildSensorDisplay(
                  'GYROSCOPE (rad/s)',
                  ['X: $gyroX', 'Y: $gyroY', 'Z: $gyroZ'],
                  const Color(0xFF10B981),
                  [data?.gyroscopeX, data?.gyroscopeY, data?.gyroscopeZ],
                ),
              ),
            ],
          )
        ],
      ),
    );
  }

  Widget _buildSensorDisplay(String title, List<String> readings, Color color, List<double?> rawVals) {
    return Container(
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: Colors.black.withOpacity(0.2),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: Colors.white.withOpacity(0.04)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          Text(
            title,
            style: TextStyle(fontSize: 10, fontWeight: FontWeight.bold, color: color, letterSpacing: 0.5),
          ),
          const SizedBox(height: 8),
          ...List.generate(readings.length, (index) {
            final val = rawVals[index] ?? 0.0;
            // Map typical sensor range into progress bar representation (cap at 15 for acc, 6 for gyro)
            final maxLimit = title.contains('ACCELEROMETER') ? 15.0 : 6.0;
            final fraction = (val.abs() / maxLimit).clamp(0.0, 1.0);

            return Padding(
              padding: const EdgeInsets.only(bottom: 6.0),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.stretch,
                children: [
                  Text(
                    readings[index],
                    style: const TextStyle(fontSize: 11, fontFamily: 'monospace', color: Colors.white70),
                  ),
                  const SizedBox(height: 2),
                  ClipRRect(
                    borderRadius: BorderRadius.circular(4),
                    child: LinearProgressIndicator(
                      value: fraction,
                      backgroundColor: Colors.white10,
                      valueColor: AlwaysStoppedAnimation<Color>(color.withOpacity(0.8)),
                      minHeight: 3,
                    ),
                  ),
                ],
              ),
            );
          }),
        ],
      ),
    );
  }

  Widget _buildInteractionCard(SensorProvider provider) {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: const Color(0xFF181824),
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: Colors.white.withOpacity(0.06)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Row(
                children: [
                  const Icon(Icons.gesture, color: Color(0xFF6366F1)),
                  const SizedBox(width: 8),
                  const Text(
                    'Biometric Gesture Canvas',
                    style: TextStyle(fontWeight: FontWeight.bold, fontSize: 15, color: Colors.white),
                  ),
                ],
              ),
              Container(
                padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                decoration: BoxDecoration(
                  color: Colors.white.withOpacity(0.05),
                  borderRadius: BorderRadius.circular(8),
                ),
                child: Text(
                  'Freq: ${provider.interactionFrequency.toStringAsFixed(0)} int/min',
                  style: const TextStyle(fontSize: 10, fontWeight: FontWeight.bold, color: Colors.amberAccent),
                ),
              ),
            ],
          ),
          const SizedBox(height: 8),
          Text(
            'Draw or Tap in the zone below to record touch coordinates, durations, velocities, and swipe trajectories.',
            style: TextStyle(color: Colors.white.withOpacity(0.5), fontSize: 11),
          ),
          const SizedBox(height: 12),
          GestureCaptureCanvas(provider: provider),
        ],
      ),
    );
  }

  Widget _buildTypingCard(SensorProvider provider) {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: const Color(0xFF181824),
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: Colors.white.withOpacity(0.06)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Row(
                children: [
                  const Icon(Icons.keyboard_outlined, color: Color(0xFF6366F1)),
                  const SizedBox(width: 8),
                  const Text(
                    'Keystroke Dynamics Capture',
                    style: TextStyle(fontWeight: FontWeight.bold, fontSize: 15, color: Colors.white),
                  ),
                ],
              ),
              Container(
                padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                decoration: BoxDecoration(
                  color: Colors.white.withOpacity(0.05),
                  borderRadius: BorderRadius.circular(8),
                ),
                child: Text(
                  'Speed: ${provider.typingSpeedWpm.toStringAsFixed(1)} WPM',
                  style: const TextStyle(fontSize: 10, fontWeight: FontWeight.bold, color: Color(0xFF10B981)),
                ),
              ),
            ],
          ),
          const SizedBox(height: 12),
          TypingCaptureZone(provider: provider),
        ],
      ),
    );
  }

  Widget _buildLogControlCard(BuildContext context, SensorProvider provider) {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: const Color(0xFF181824),
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: Colors.white.withOpacity(0.06)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Row(
                children: [
                  const Icon(Icons.analytics_outlined, color: Color(0xFF6366F1)),
                  const SizedBox(width: 8),
                  const Text(
                    'ML Training Exporter',
                    style: TextStyle(fontWeight: FontWeight.bold, fontSize: 15, color: Colors.white),
                  ),
                ],
              ),
              Container(
                padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
                decoration: BoxDecoration(
                  color: Colors.indigoAccent.withOpacity(0.15),
                  borderRadius: BorderRadius.circular(12),
                  border: Border.all(color: Colors.indigoAccent.withOpacity(0.3)),
                ),
                child: Text(
                  '${provider.dataCount} snapshots',
                  style: const TextStyle(fontSize: 11, fontWeight: FontWeight.bold, color: Colors.indigoAccent),
                ),
              ),
            ],
          ),
          const SizedBox(height: 12),
          Row(
            children: [
              Expanded(
                child: OutlinedButton.icon(
                  onPressed: provider.dataCount == 0 ? null : () => _showExportDataDialog(context, provider, 'CSV'),
                  icon: const Icon(Icons.table_view, size: 16),
                  label: const Text('Export CSV'),
                  style: OutlinedButton.styleFrom(
                    foregroundColor: const Color(0xFF10B981),
                    side: BorderSide(color: const Color(0xFF10B981).withOpacity(0.4)),
                    shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(10)),
                  ),
                ),
              ),
              const SizedBox(width: 10),
              Expanded(
                child: OutlinedButton.icon(
                  onPressed: provider.dataCount == 0 ? null : () => _showExportDataDialog(context, provider, 'JSON'),
                  icon: const Icon(Icons.code, size: 16),
                  label: const Text('Export JSON'),
                  style: OutlinedButton.styleFrom(
                    foregroundColor: Colors.indigoAccent,
                    side: BorderSide(color: Colors.indigoAccent.withOpacity(0.4)),
                    shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(10)),
                  ),
                ),
              ),
            ],
          ),
          const SizedBox(height: 8),
          TextButton.icon(
            onPressed: provider.dataCount == 0
                ? null
                : () {
                    provider.clearData();
                    ScaffoldMessenger.of(context).showSnackBar(
                      const SnackBar(content: Text('Memory buffer cleared.')),
                    );
                  },
            icon: const Icon(Icons.delete_outline, size: 16),
            label: const Text('Clear Buffer'),
            style: TextButton.styleFrom(
              foregroundColor: Colors.redAccent.withOpacity(0.8),
            ),
          ),
          if (provider.collectedData.isNotEmpty) ...[
            const Divider(color: Colors.white10, height: 20),
            Text(
              'LATEST BUFFER RECORDS (FIFO)',
              style: TextStyle(fontWeight: FontWeight.bold, fontSize: 10, color: Colors.white.withOpacity(0.4)),
            ),
            const SizedBox(height: 8),
            ListView.builder(
              shrinkWrap: true,
              physics: const NeverScrollableScrollPhysics(),
              itemCount: min(provider.collectedData.length, 3),
              itemBuilder: (context, index) {
                // Read from reverse (newest first)
                final revIdx = provider.collectedData.length - 1 - index;
                final item = provider.collectedData[revIdx];
                final date = DateTime.fromMillisecondsSinceEpoch(item.timestamp);
                final timeStr = '${date.hour.toString().padLeft(2, '0')}:${date.minute.toString().padLeft(2, '0')}:${date.second.toString().padLeft(2, '0')}.${(date.millisecond).toString().padLeft(3, '0')}';

                String detail = 'Sensors only';
                if (item.touchX != null && item.swipeSpeed != null) {
                  detail = 'Swipe: ${item.swipeDirection} | Spd: ${item.swipeSpeed!.toStringAsFixed(2)} px/ms';
                } else if (item.touchX != null && item.tapDurationMs != null) {
                  detail = 'Tap: (${item.touchX!.toStringAsFixed(0)}, ${item.touchY!.toStringAsFixed(0)}) | Dur: ${item.tapDurationMs}ms';
                } else if (item.keystrokeIntervalMs != null) {
                  detail = 'Keystroke interval: ${item.keystrokeIntervalMs}ms';
                }

                return Padding(
                  padding: const EdgeInsets.only(bottom: 6.0),
                  child: Container(
                    padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 8),
                    decoration: BoxDecoration(
                      color: Colors.white.withOpacity(0.03),
                      borderRadius: BorderRadius.circular(8),
                      border: Border.all(color: Colors.white.withOpacity(0.03)),
                    ),
                    child: Row(
                      mainAxisAlignment: MainAxisAlignment.spaceBetween,
                      children: [
                        Text(
                          timeStr,
                          style: const TextStyle(fontSize: 10, fontFamily: 'monospace', color: Colors.indigoAccent),
                        ),
                        Expanded(
                          child: Text(
                            detail,
                            style: const TextStyle(fontSize: 11, color: Colors.white70),
                            textAlign: TextAlign.end,
                            overflow: TextOverflow.ellipsis,
                          ),
                        ),
                      ],
                    ),
                  ),
                );
              },
            ),
          ]
        ],
      ),
    );
  }

  void _showExportDataDialog(BuildContext context, SensorProvider provider, String format) {
    final text = format == 'CSV' ? provider.exportToCsv() : provider.exportToJson();

    showDialog(
      context: context,
      builder: (context) {
        return AlertDialog(
          backgroundColor: const Color(0xFF1E1E2F),
          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
          title: Text('Exported Biometrics ($format)'),
          content: SizedBox(
            width: double.maxFinite,
            height: 300,
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.stretch,
              children: [
                Text(
                  'Format ready for ML Training models (length: ${text.length} chars).',
                  style: TextStyle(color: Colors.white.withOpacity(0.5), fontSize: 12),
                ),
                const SizedBox(height: 10),
                Expanded(
                  child: Container(
                    padding: const EdgeInsets.all(10),
                    decoration: BoxDecoration(
                      color: Colors.black38,
                      borderRadius: BorderRadius.circular(10),
                      border: Border.all(color: Colors.white.withOpacity(0.08)),
                    ),
                    child: SingleChildScrollView(
                      physics: const BouncingScrollPhysics(),
                      child: SelectableText(
                        text,
                        style: const TextStyle(fontFamily: 'monospace', fontSize: 10, color: Colors.white70),
                      ),
                    ),
                  ),
                ),
              ],
            ),
          ),
          actions: [
            TextButton(
              onPressed: () => Navigator.of(context).pop(),
              child: const Text('Close'),
            ),
            ElevatedButton.icon(
              onPressed: () {
                Clipboard.setData(ClipboardData(text: text));
                ScaffoldMessenger.of(context).showSnackBar(
                  SnackBar(content: Text('Copied $format data to clipboard.')),
                );
                Navigator.of(context).pop();
              },
              icon: const Icon(Icons.copy, size: 14),
              label: const Text('Copy All'),
              style: ElevatedButton.styleFrom(
                backgroundColor: const Color(0xFF6366F1),
                foregroundColor: Colors.white,
              ),
            ),
          ],
        );
      },
    );
  }

  void _showExplainDialog(BuildContext context) {
    showDialog(
      context: context,
      builder: (context) {
        return AlertDialog(
          backgroundColor: const Color(0xFF1E1E2F),
          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
          title: const Row(
            children: [
              Icon(Icons.info, color: Colors.indigoAccent),
              SizedBox(width: 10),
              Text('Security Biometrics Info'),
            ],
          ),
          content: const SingleChildScrollView(
            physics: BouncingScrollPhysics(),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                _ExplainSection(
                  title: '1. Accelerometer & Gyroscope',
                  body: 'Used to log raw hand jitter, motion angles, and device tilt. Unique patterns in holding angles serve as a vital biometric signature.',
                ),
                _ExplainSection(
                  title: '2. Device Orientation',
                  body: 'Resolves landscape/portrait context to scale and normalize coordinates, ensuring ML models interpret positioning accurately.',
                ),
                _ExplainSection(
                  title: '3. Touch & Swipe Speed',
                  body: 'Tracks user swipe velocity (pixels/ms). Calculated by dividing geometric swipe distance by pointer travel duration.',
                ),
                _ExplainSection(
                  title: '4. Swipe Trajectory & Angle',
                  body: 'Calculates swipe distance in pixels and direction/angle in degrees using vector mechanics (dx, dy coordinates).',
                ),
                _ExplainSection(
                  title: '5. Tap Duration',
                  body: 'Measured as the time delta (milliseconds) between pointer down and pointer up transitions. Varies characteristically per user.',
                ),
                _ExplainSection(
                  title: '6. Typing Speed (Keystroke Dynamics)',
                  body: 'Estimates WPM and tracks keyflight time latency (interval delay between keys) to create unique typing fingerprints.',
                ),
              ],
            ),
          ),
          actions: [
            TextButton(
              onPressed: () => Navigator.of(context).pop(),
              child: const Text('Got it'),
            ),
          ],
        );
      },
    );
  }
}

class _ExplainSection extends StatelessWidget {
  final String title;
  final String body;

  const _ExplainSection({required this.title, required this.body});

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 12.0),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(title, style: const TextStyle(fontWeight: FontWeight.bold, fontSize: 13, color: Colors.indigoAccent)),
          const SizedBox(height: 4),
          Text(body, style: TextStyle(fontSize: 12, color: Colors.white.withOpacity(0.8))),
        ],
      ),
    );
  }
}

class GestureCaptureCanvas extends StatefulWidget {
  final SensorProvider provider;
  const GestureCaptureCanvas({super.key, required this.provider});

  @override
  State<GestureCaptureCanvas> createState() => _GestureCaptureCanvasState();
}

class _GestureCaptureCanvasState extends State<GestureCaptureCanvas> {
  Offset? _startPos;
  DateTime? _startTime;
  Offset? _currentPos;
  String _lastGestureInfo = "Perform a tap or swipe in this canvas area";
  final List<Offset> _trail = [];

  void _onPointerDown(PointerDownEvent event) {
    setState(() {
      _startPos = event.localPosition;
      _startTime = DateTime.now();
      _currentPos = event.localPosition;
      _trail.clear();
      _trail.add(event.localPosition);
    });
  }

  void _onPointerMove(PointerMoveEvent event) {
    setState(() {
      _currentPos = event.localPosition;
      _trail.add(event.localPosition);
    });
  }

  void _onPointerUp(PointerUpEvent event) {
    if (_startPos == null || _startTime == null) return;
    
    final endPos = event.localPosition;
    final endTime = DateTime.now();
    final durationMs = endTime.difference(_startTime!).inMilliseconds;
    final delta = endPos - _startPos!;
    final distance = delta.distance;

    if (durationMs <= 0) return;

    if (distance < 15.0) {
      // Classified as a Tap
      widget.provider.registerTap(
        x: _startPos!.dx,
        y: _startPos!.dy,
        durationMs: durationMs,
      );
      setState(() {
        _lastGestureInfo = "📥 TAP REGISTERED\n"
            "• Coordinates: (${_startPos!.dx.toStringAsFixed(1)}, ${_startPos!.dy.toStringAsFixed(1)})\n"
            "• Tap Hold Time: ${durationMs}ms";
        _trail.clear();
        _startPos = null;
      });
    } else {
      // Classified as a Swipe
      final speed = distance / durationMs; // pixels per ms
      final angleRad = atan2(delta.dy, delta.dx);
      final angleDeg = (angleRad * 180 / pi + 360) % 360;
      
      // Determine dominant direction
      String direction = "None";
      if (delta.dx.abs() > delta.dy.abs()) {
        direction = delta.dx > 0 ? "Right" : "Left";
      } else {
        direction = delta.dy > 0 ? "Down" : "Up";
      }
      final directionString = "$direction (${angleDeg.toStringAsFixed(0)}°)";

      widget.provider.registerSwipe(
        startX: _startPos!.dx,
        startY: _startPos!.dy,
        distance: distance,
        direction: directionString,
        speed: speed,
        durationMs: durationMs,
      );

      setState(() {
        _lastGestureInfo = "📥 SWIPE REGISTERED\n"
            "• Start: (${_startPos!.dx.toStringAsFixed(1)}, ${_startPos!.dy.toStringAsFixed(1)})\n"
            "• Distance: ${distance.toStringAsFixed(1)} px\n"
            "• Direction: $directionString\n"
            "• Velocity: ${speed.toStringAsFixed(3)} px/ms\n"
            "• Duration: ${durationMs}ms";
        _trail.clear();
        _startPos = null;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        ClipRRect(
          borderRadius: BorderRadius.circular(12),
          child: Container(
            height: 180,
            color: Colors.black45,
            child: Listener(
              onPointerDown: _onPointerDown,
              onPointerMove: _onPointerMove,
              onPointerUp: _onPointerUp,
              child: CustomPaint(
                painter: _GestureTrailPainter(_trail, _startPos, _currentPos),
                child: const Center(
                  child: Icon(Icons.touch_app_outlined, color: Colors.white24, size: 40),
                ),
              ),
            ),
          ),
        ),
        const SizedBox(height: 10),
        Container(
          padding: const EdgeInsets.all(12),
          decoration: BoxDecoration(
            color: Colors.white.withOpacity(0.02),
            borderRadius: BorderRadius.circular(10),
            border: Border.all(color: Colors.white.withOpacity(0.04)),
          ),
          child: Text(
            _lastGestureInfo,
            style: const TextStyle(
              fontSize: 11,
              fontFamily: 'monospace',
              color: Colors.white70,
              height: 1.4,
            ),
          ),
        )
      ],
    );
  }
}

class _GestureTrailPainter extends CustomPainter {
  final List<Offset> trail;
  final Offset? start;
  final Offset? current;

  _GestureTrailPainter(this.trail, this.start, this.current);

  @override
  void paint(Canvas canvas, Size size) {
    if (trail.isEmpty) return;

    final paint = Paint()
      ..color = const Color(0xFF6366F1).withOpacity(0.6)
      ..strokeWidth = 4.0
      ..strokeCap = StrokeCap.round
      ..style = PaintingStyle.stroke;

    final path = Path();
    path.moveTo(trail.first.dx, trail.first.dy);
    for (int i = 1; i < trail.length; i++) {
      path.lineTo(trail[i].dx, trail[i].dy);
    }
    canvas.drawPath(path, paint);

    // Draw touch points
    if (start != null) {
      canvas.drawCircle(start!, 6.0, Paint()..color = const Color(0xFF10B981));
    }
    if (current != null) {
      canvas.drawCircle(current!, 4.0, Paint()..color = const Color(0xFFEF4444));
    }
  }

  @override
  bool shouldRepaint(covariant _GestureTrailPainter oldDelegate) => true;
}

class TypingCaptureZone extends StatefulWidget {
  final SensorProvider provider;
  const TypingCaptureZone({super.key, required this.provider});

  @override
  State<TypingCaptureZone> createState() => _TypingCaptureZoneState();
}

class _TypingCaptureZoneState extends State<TypingCaptureZone> {
  final TextEditingController _controller = TextEditingController();
  int _lastTextLength = 0;

  @override
  void initState() {
    super.initState();
    _controller.addListener(_onTextChanged);
  }

  void _onTextChanged() {
    final text = _controller.text;
    final currentLength = text.length;

    // Detect forward typing to capture keyboard entry
    if (currentLength > _lastTextLength) {
      widget.provider.registerKeystroke();
    }
    _lastTextLength = currentLength;
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        KeyboardListener(
          focusNode: FocusNode(), // Simple dummy focus node
          onKeyEvent: (event) {
            // Optional capture point for hardware keyboards
            if (event is KeyDownEvent) {
              // Simply record keystroke timing
            }
          },
          child: TextField(
            controller: _controller,
            maxLines: 2,
            style: const TextStyle(color: Colors.white, fontSize: 13, fontFamily: 'sans-serif'),
            decoration: InputDecoration(
              hintText: 'Type text here to evaluate keyboard biometrics...',
              hintStyle: TextStyle(color: Colors.white.withOpacity(0.3), fontSize: 12),
              filled: true,
              fillColor: Colors.black26,
              border: OutlineInputBorder(
                borderRadius: BorderRadius.circular(12),
                borderSide: BorderSide.none,
              ),
              enabledBorder: OutlineInputBorder(
                borderRadius: BorderRadius.circular(12),
                borderSide: BorderSide(color: Colors.white.withOpacity(0.06)),
              ),
              focusedBorder: OutlineInputBorder(
                borderRadius: BorderRadius.circular(12),
                borderSide: const BorderSide(color: Color(0xFF6366F1), width: 1.5),
              ),
              contentPadding: const EdgeInsets.symmetric(horizontal: 14, vertical: 12),
            ),
          ),
        ),
        const SizedBox(height: 6),
        Row(
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          children: [
            Text(
              'Length: ${_controller.text.length} chars',
              style: TextStyle(color: Colors.white.withOpacity(0.4), fontSize: 10),
            ),
            GestureDetector(
              onTap: () {
                _controller.clear();
                setState(() => _lastTextLength = 0);
              },
              child: const Text(
                'Clear Field',
                style: TextStyle(color: Colors.indigoAccent, fontSize: 11, fontWeight: FontWeight.bold),
              ),
            )
          ],
        )
      ],
    );
  }
}
