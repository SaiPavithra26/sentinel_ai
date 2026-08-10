import 'dart:developer' as developer;

/// Categorizes the level of security risk detected by the behavioral biometrics.
enum RiskLevel {
  /// Risk score 0–30: The behavior is consistent with the legitimate owner.
  safe,

  /// Risk score 31–60: Minor behavioral deviations or minor sensor anomalies.
  warning,

  /// Risk score 61–80: Significant behavioral mismatch, warning of potential intrusion.
  suspicious,

  /// Risk score 81–100: Extreme anomaly indicators, critical risk of active unauthorized access.
  critical,
}

/// Extension on [RiskLevel] to provide human-readable labels and descriptions.
extension RiskLevelExtensions on RiskLevel {
  String get name {
    switch (this) {
      case RiskLevel.safe:
        return 'Safe';
      case RiskLevel.warning:
        return 'Warning';
      case RiskLevel.suspicious:
        return 'Suspicious';
      case RiskLevel.critical:
        return 'Critical';
    }
  }

  String get description {
    switch (this) {
      case RiskLevel.safe:
        return 'Normal behavioral profile. The user is authenticated as the owner.';
      case RiskLevel.warning:
        return 'Minor deviation from the behavioral profile. Continued monitoring recommended.';
      case RiskLevel.suspicious:
        return 'High deviation from the behavioral profile. High risk of unauthorized user.';
      case RiskLevel.critical:
        return 'Critical profile mismatch. The user is flagged as highly suspicious.';
    }
  }
}

/// Represents the structured security evaluation report produced by the [RiskEngine].
class RiskReport {
  /// The combined risk score, ranged from 0.0 to 100.0.
  final double riskScore;

  /// The categorized risk level (Safe, Warning, Suspicious, Critical).
  final RiskLevel riskLevel;

  /// Map showing the individual weighted contributions to the final risk score.
  final Map<String, double> contributions;

  /// Map showing the raw, unweighted input scores.
  final Map<String, double> rawInputs;

  /// The timestamp of when this risk report was calculated.
  final DateTime timestamp;

  RiskReport({
    required this.riskScore,
    required this.riskLevel,
    required this.contributions,
    required this.rawInputs,
    required this.timestamp,
  });

  @override
  String toString() {
    return 'RiskReport(Score: ${riskScore.toStringAsFixed(2)}, Level: ${riskLevel.name}, Time: $timestamp)';
  }
}

/// A production-ready engine that computes a unified risk score from 0 to 100
/// based on multiple behavioral indicators.
/// 
/// ### SCORING FORMULA & MATHEMATICAL DESIGN
/// 
/// The risk score combines predictions from deep learning classifier (TensorFlow Lite)
/// and heuristic anomaly detection models across multiple behavioral modalities.
/// 
/// Let $M = \{\text{TF}, \text{sensor}, \text{typing}, \text{swipe}, \text{movement}\}$ be the set of risk indicators.
/// Let $w_m$ be the base weight of metric $m$:
/// - $w_{\text{TF}} = 0.50$ (Deep learning verification carries the primary weight)
/// - $w_{\text{typing}} = 0.15$ (Keystroke velocity and pause anomalies)
/// - $w_{\text{swipe}} = 0.15$ (Swipe duration, speed, and angle anomalies)
/// - $w_{\text{movement}} = 0.15$ (Accelerometer/Gyroscope statistical anomalies)
/// - $w_{\text{sensor}} = 0.05$ (General ambient sensor anomalies)
/// 
/// To support real-time data collection where some channels may be temporarily inactive
/// (e.g., no keyboard inputs during a map reading session), the engine uses a **Dynamic weight redistribution**.
/// 
/// Let $I_m \in \{0, 1\}$ be an indicator variable where:
/// - $I_m = 1$ if indicator $m$ is available (not null).
/// - $I_m = 0$ if indicator $m$ is unavailable (null).
/// 
/// The total weight of all active indicators is calculated as:
/// $$W_{\text{total}} = \sum_{m \in M} w_m \cdot I_m$$
/// 
/// The normalized weight for each active indicator $\hat{w}_m$ is computed to guarantee the sum equals 1.0:
/// $$\hat{w}_m = \frac{w_m}{W_{\text{total}}}$$
/// 
/// The risk contribution $r_m \in [0, 100]$ of each indicator is defined as:
/// - $r_{\text{TF}} = (1.0 - \text{ownerConfidence}) \times 100$
/// - $r_{\text{typing}} = \text{typingAnomaly} \times 100$
/// - $r_{\text{swipe}} = \text{swipeAnomaly} \times 100$
/// - $r_{\text{movement}} = \text{movementAnomaly} \times 100$
/// - $r_{\text{sensor}} = \text{sensorAnomaly} \times 100$
/// 
/// The unified risk score $R \in [0, 100]$ is the weighted sum of active risk contributions:
/// $$R = \sum_{m \in M} I_m \cdot \hat{w}_m \cdot r_m$$
class RiskEngine {
  // Base weights for the risk indicators
  final double _weightTF;
  final double _weightSensor;
  final double _weightTyping;
  final double _weightSwipe;
  final double _weightMovement;

  /// Default constructor with standard behavioral biometrics weights.
  RiskEngine({
    double weightTF = 0.50,
    double weightSensor = 0.05,
    double weightTyping = 0.15,
    double weightSwipe = 0.15,
    double weightMovement = 0.15,
  })  : _weightTF = weightTF,
        _weightSensor = weightSensor,
        _weightTyping = weightTyping,
        _weightSwipe = weightSwipe,
        _weightMovement = weightMovement {
    // Assert that the base weights sum to 1.0 to prevent scaling errors
    final total = _weightTF + _weightSensor + _weightTyping + _weightSwipe + _weightMovement;
    assert((total - 1.0).abs() < 1e-9, 'Base weights must sum to exactly 1.0 (Sum: $total)');
  }

  /// Calculates a unified [RiskReport] based on the active behavioral indicators.
  /// 
  /// - [tfOwnerConfidence]: Probability confidence (0.0 to 1.0) that the user is the legitimate owner.
  /// - [sensorAnomaly]: Anomaly score (0.0 to 1.0) for general environmental sensors.
  /// - [typingAnomaly]: Anomaly score (0.0 to 1.0) for keystroke dynamics.
  /// - [swipeAnomaly]: Anomaly score (0.0 to 1.0) for touch gestures.
  /// - [movementAnomaly]: Anomaly score (0.0 to 1.0) for physical movement (accelerometer/gyroscope).
  RiskReport calculateRisk({
    double? tfOwnerConfidence,
    double? sensorAnomaly,
    double? typingAnomaly,
    double? swipeAnomaly,
    double? movementAnomaly,
  }) {
    final DateTime now = DateTime.now();

    // Map input fields to their base weights and indicator functions
    final Map<String, _RiskMetricInput> inputs = {
      'tf': _RiskMetricInput(
        value: tfOwnerConfidence != null ? 1.0 - tfOwnerConfidence : null,
        baseWeight: _weightTF,
      ),
      'sensor': _RiskMetricInput(value: sensorAnomaly, baseWeight: _weightSensor),
      'typing': _RiskMetricInput(value: typingAnomaly, baseWeight: _weightTyping),
      'swipe': _RiskMetricInput(value: swipeAnomaly, baseWeight: _weightSwipe),
      'movement': _RiskMetricInput(value: movementAnomaly, baseWeight: _weightMovement),
    };

    double activeWeightSum = 0.0;
    inputs.forEach((key, metric) {
      if (metric.isActive) {
        activeWeightSum += metric.baseWeight;
      }
    });

    // In case no indicators are available, fall back to safe baseline (0.0)
    if (activeWeightSum <= 0.0) {
      developer.log(
        'RiskEngine: No active behavioral metrics provided. Defaulting to safe risk score.',
        name: 'RiskEngine',
      );
      return RiskReport(
        riskScore: 0.0,
        riskLevel: RiskLevel.safe,
        contributions: {},
        rawInputs: {},
        timestamp: now,
      );
    }

    double finalScoreSum = 0.0;
    final Map<String, double> contributions = {};
    final Map<String, double> rawInputs = {};

    inputs.forEach((key, metric) {
      if (metric.isActive) {
        // Normalize weight relative to the sum of all active weights
        final normalizedWeight = metric.baseWeight / activeWeightSum;
        final rawScore = metric.value! * 100.0;
        final weightedContribution = rawScore * normalizedWeight;

        finalScoreSum += weightedContribution;
        contributions[key] = weightedContribution;
        rawInputs[key] = rawScore;
      }
    });

    // Clamp final score strictly in range [0.0, 100.0] to prevent floating point anomalies
    final double finalRiskScore = finalScoreSum.clamp(0.0, 100.0);
    final RiskLevel calculatedLevel = _determineRiskLevel(finalRiskScore);

    developer.log(
      'RiskEngine: Risk calculated: ${finalRiskScore.toStringAsFixed(2)}% -> Level: ${calculatedLevel.name}',
      name: 'RiskEngine',
    );

    return RiskReport(
      riskScore: finalRiskScore,
      riskLevel: calculatedLevel,
      contributions: contributions,
      rawInputs: rawInputs,
      timestamp: now,
    );
  }

  /// Determines the [RiskLevel] category corresponding to the calculated risk score.
  RiskLevel _determineRiskLevel(double score) {
    if (score <= 30.0) {
      return RiskLevel.safe;
    } else if (score <= 60.0) {
      return RiskLevel.warning;
    } else if (score <= 80.0) {
      return RiskLevel.suspicious;
    } else {
      return RiskLevel.critical;
    }
  }
}

/// Helper internal class to represent a metric input state.
class _RiskMetricInput {
  final double? value;
  final double baseWeight;

  _RiskMetricInput({
    required this.value,
    required this.baseWeight,
  });

  /// An indicator is active only if its value is provided and valid.
  bool get isActive => value != null && !value!.isNaN;
}
