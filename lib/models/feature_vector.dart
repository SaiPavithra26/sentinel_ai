import 'behavior_data.dart';
import '../services/feature_extractor.dart';

/// Represents a combined and preprocessed behavioral biometric feature vector.
/// 
/// ### MATHEMATICAL APPROACH
/// 
/// This class combines multi-modal biometric signals (accelerometer, gyroscope,
/// device orientation, touch taps, swipe gestures, and typing dynamics) into a single,
/// unified numerical representation suitable for real-time model inference.
/// 
/// The mathematical pipeline consists of two phases:
/// 
/// #### Phase 1: Statistical Feature Extraction
/// 1. **Motion Dynamics (Accelerometer & Gyroscope):**
///    Aggregated streams are split into coordinate axes (X, Y, Z) and Magnitude:
///    $$Mag_t = \sqrt{X_t^2 + Y_t^2 + Z_t^2}$$
///    For each sensor column, we calculate 7 statistical moments:
///    - **Mean ($\mu$):** Central location of the signal.
///      $$\mu = \frac{1}{N}\sum_{i=1}^N x_i$$
///    - **Standard Deviation ($\sigma$):** Dispersion of sensor readings.
///      $$\sigma = \sqrt{\frac{1}{N-1}\sum_{i=1}^N (x_i - \mu)^2}$$
///    - **Min/Max/Median:** Structural limits and middle percentile.
///    - **Skewness ($\gamma_1$):** Lack of symmetry in motion distribution.
///      $$\gamma_1 = \frac{N}{(N-1)(N-2)} \sum_{i=1}^N \left(\frac{x_i - \mu}{\sigma}\right)^3$$
///    - **Kurtosis ($\beta_2$):** Measure of outliers and peak tails (heavy/light motion).
///      $$\beta_2 = \frac{N(N+1)}{(N-1)(N-2)(N-3)} \sum_{i=1}^N \left(\frac{x_i - \mu}{\sigma}\right)^4 - \frac{3(N-1)^2}{(N-2)(N-3)}$$
/// 
/// 2. **Touch & Swipe Mechanics:**
///    - **Euclidean Swipe Distance ($D$):**
///      $$D = \sqrt{(X_{end} - X_{start})^2 + (Y_{end} - Y_{start})^2}$$
///    - **Swipe Duration ($T$):** Total swipe execution time in seconds.
///    - **Swipe Speed ($S$):** $S = D / T$.
///    - **Swipe Direction / Angle ($\theta$):**
///      $$\theta = \text{atan2}(Y_{end} - Y_{start}, X_{end} - X_{start})$$
///      To avoid wrap-around errors at $\pm\pi$, circular statistics are used:
///      - Circular Mean: $\bar{\theta} = \text{atan2}\left(\frac{1}{M}\sum \sin \theta_i, \frac{1}{M}\sum \cos \theta_i\right)$
///      - Circular Std: $\sigma_\theta = \sqrt{-2 \ln(R)}$, where $R = \sqrt{(\frac{1}{M}\sum \sin\theta_i)^2 + (\frac{1}{M}\sum \cos\theta_i)^2}$
///    - **Taps and Metadata:** Duration of simple taps, average touch pressures, contact area, and finger orientation statistics.
/// 
/// 3. **Keystroke Dynamics:**
///    - **Typing Speed ($W$):** Keystroke count normalized by duration.
///    - **Key Hold Times:** Interval between press and release.
///    - **Flight Times ($F_1, F_2, F_3, F_4$):** Latency intervals between consecutive keys (Down-Up, Up-Down, Down-Down, Up-Up).
/// 
/// #### Phase 2: Mathematical Preprocessing & Standardization
/// Once a raw feature map is compiled, it is aligned and transformed into a unified feature vector:
/// 
/// 1. **Outlier Clipping:**
///    Protects the inference engine from sensor spikes by clamping values:
///    $$x_{clipped} = \min(\text{upper\_bound}, \max(\text{lower\_bound}, x))$$
/// 
/// 2. **Median Imputation:**
///    Fills missing values (e.g. no typing features during navigation tasks) with training-set medians:
///    $$x_{imputed} = \text{isNaN}(x) ? \tilde{x}_{median} : x$$
/// 
/// 3. **MinMax Normalization:**
///    Rescales features linearly to the training interval bounds ($[0, 1]$):
///    $$x_{minmax} = x_{imputed} \cdot \text{scale}_{minmax} + \text{min}_{minmax}$$
/// 
/// 4. **Feature Selection & Ordering:**
///    Extracts only the top $K$ features (typically 50) verified by variance thresholds, correlation filters, and ANOVA:
///    $$\vec{v}_{selected} = [x_{minmax}^{(selected\_1)}, x_{minmax}^{(selected\_2)}, \dots, x_{minmax}^{(selected\_K)}]$$
/// 
/// 5. **Standardization (StandardScaler):**
///    Centers and scales the selected features to zero mean ($\mu = 0$) and unit variance ($\sigma = 1$):
///    $$x_{standardized} = \frac{x_{minmax} - \mu_{train}}{\sigma_{train}}$$
class BehaviorFeatureVector {
  /// The raw, unscaled feature map containing all statistical moments.
  final Map<String, double> rawFeatures;

  /// The final, aligned, clipped, imputed, scaled, and standardized feature vector ready for ML.
  final List<double> preprocessedVector;

  /// Metadata specifying when the feature vector was generated.
  final DateTime timestamp;

  /// The target owner ID for whom this vector is prepared.
  final String ownerId;

  BehaviorFeatureVector({
    required this.rawFeatures,
    required this.preprocessedVector,
    required this.timestamp,
    required this.ownerId,
  });

  /// Factory constructor that combines raw sensor data points, extracts statistical features,
  /// and applies the mathematical preprocessing pipeline using configuration parameters.
  factory BehaviorFeatureVector.fromRawData(
    List<BehaviorData> sessionData, {
    required Map<String, dynamic> preprocessParams,
    String ownerId = '1',
  }) {
    // 1. Run statistical feature extraction
    final Map<String, double> rawFeatures = FeatureExtractor.extractFeatures(
      sessionData,
      sessionDurationMins: 5.0,
    );

    // Load pipeline parameters from preprocessing JSON config
    final featureCols = List<String>.from(preprocessParams['feature_cols'] as List);
    final clippingBounds = preprocessParams['clipping_bounds'] as Map<String, dynamic>;
    final imputerMedians = preprocessParams['imputer_medians'] as Map<String, dynamic>;
    final minmaxMin = preprocessParams['minmax_min'] as Map<String, dynamic>;
    final minmaxScale = preprocessParams['minmax_scale'] as Map<String, dynamic>;

    final selectedFeatures = List<String>.from(preprocessParams['selected_features'] as List);
    final standardMean = preprocessParams['standard_mean'] as Map<String, dynamic>;
    final standardScale = preprocessParams['standard_scale'] as Map<String, dynamic>;

    // Initialize list matching original training column alignment
    final List<double> alignedVector = List<double>.filled(featureCols.length, 0.0);

    for (int i = 0; i < featureCols.length; i++) {
      final col = featureCols[i];
      double val = rawFeatures[col] ?? double.nan;

      // Mathematical Step 1: Outlier Clipping
      if (clippingBounds.containsKey(col)) {
        final bounds = clippingBounds[col] as Map<String, dynamic>;
        final lower = (bounds['lower'] as num).toDouble();
        final upper = (bounds['upper'] as num).toDouble();
        if (!val.isNaN) {
          val = val.clamp(lower, upper);
        }
      }

      // Mathematical Step 2: Median Imputation
      if (val.isNaN) {
        val = (imputerMedians[col] as num).toDouble();
      }

      // Mathematical Step 3: MinMaxScaler Scaling
      final minVal = (minmaxMin[col] as num).toDouble();
      final scaleVal = (minmaxScale[col] as num).toDouble();
      val = val * scaleVal + minVal;

      alignedVector[i] = val;
    }

    // Mathematical Step 4 & 5: Feature Selection & StandardScaler
    final List<double> preprocessedVector = List<double>.filled(selectedFeatures.length, 0.0);

    for (int i = 0; i < selectedFeatures.length; i++) {
      final col = selectedFeatures[i];
      final originalIndex = featureCols.indexOf(col);

      // Extract value from aligned vector (default to 0.0 if not found)
      double val = (originalIndex != -1) ? alignedVector[originalIndex] : 0.0;

      // Apply StandardScaler transformation
      final meanVal = (standardMean[col] as num).toDouble();
      final stdVal = (standardScale[col] as num).toDouble();

      // Avoid division by zero
      val = (val - meanVal) / (stdVal > 1e-9 ? stdVal : 1.0);
      preprocessedVector[i] = val;
    }

    return BehaviorFeatureVector(
      rawFeatures: rawFeatures,
      preprocessedVector: preprocessedVector,
      timestamp: DateTime.now(),
      ownerId: ownerId,
    );
  }

  /// Converts the feature vector to a JSON-compatible representation.
  Map<String, dynamic> toJson() {
    return {
      'timestamp': timestamp.toIso8601String(),
      'ownerId': ownerId,
      'preprocessedVectorLength': preprocessedVector.length,
      'preprocessedVector': preprocessedVector,
      'rawFeaturesCount': rawFeatures.length,
    };
  }
}
