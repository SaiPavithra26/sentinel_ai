import 'dart:convert';
import 'package:flutter/services.dart';
import 'ml_service.dart';

/// A production-ready service to manage loading the TFLite model,
/// parsing the preprocessing parameters, applying identical data preparation,
/// and executing model inference.
class TfliteInferenceService {
  final MLService _mlService = MLService();
  Map<String, dynamic>? _preprocessParams;
  double _threshold = 0.5;
  double _eer = 0.5;
  bool _isLoaded = false;

  bool get isLoaded => _isLoaded;
  double get threshold => _threshold;
  double get eer => _eer;

  /// Loads the TFLite model, preprocessing parameters, and user-specific threshold calibration from assets.
  Future<void> initialize({String ownerId = '1'}) async {
    try {
      // 1. Load Preprocessing Configuration JSON
      final preprocessJsonStr = await rootBundle.loadString('models/preprocess_params.json');
      _preprocessParams = jsonDecode(preprocessJsonStr) as Map<String, dynamic>;

      // 2. Load User Calibration Config JSON if available
      try {
        final configJsonStr = await rootBundle.loadString('models/user_$ownerId/user_config.json');
        final userCfg = jsonDecode(configJsonStr) as Map<String, dynamic>;
        _threshold = (userCfg['threshold'] as num?)?.toDouble() ?? 0.5;
        _eer = (userCfg['eer'] as num?)?.toDouble() ?? 0.5;
      } catch (err) {
        _threshold = 0.5;
        _eer = 0.5;
      }

      // 3. Load TFLite Model
      final success = await _mlService.loadModel(ownerId: ownerId);
      _isLoaded = success;
    } catch (e) {
      _isLoaded = false;
    }
  }

  /// Takes a raw feature map extracted from Dart and applies the exact
  /// preprocessing pipeline (imputation, clipping, scaling, feature selection)
  /// before feeding to the TFLite model.
  double predict(Map<String, double> rawFeatures) {
    if (!_isLoaded || !_mlService.isModelLoaded || _preprocessParams == null) {
      return 0.5;
    }

    final featureCols = List<String>.from(_preprocessParams!['feature_cols'] as List);
    final clippingBounds = _preprocessParams!['clipping_bounds'] as Map<String, dynamic>;
    final imputerMedians = _preprocessParams!['imputer_medians'] as Map<String, dynamic>;
    final minmaxMin = _preprocessParams!['minmax_min'] as Map<String, dynamic>;
    final minmaxScale = _preprocessParams!['minmax_scale'] as Map<String, dynamic>;

    final selectedFeatures = List<String>.from(_preprocessParams!['selected_features'] as List);
    final standardMean = _preprocessParams!['standard_mean'] as Map<String, dynamic>;
    final standardScale = _preprocessParams!['standard_scale'] as Map<String, dynamic>;

    // Preprocess vector matching original train columns
    final List<double> preprocessedVector = List<double>.filled(featureCols.length, 0.0);

    for (int i = 0; i < featureCols.length; i++) {
      final col = featureCols[i];
      double val = rawFeatures[col] ?? double.nan;

      // 1. Apply outlier clipping
      if (clippingBounds.containsKey(col)) {
        final bounds = clippingBounds[col] as Map<String, dynamic>;
        final lower = (bounds['lower'] as num).toDouble();
        final upper = (bounds['upper'] as num).toDouble();
        if (!val.isNaN) {
          val = val.clamp(lower, upper);
        }
      }

      // 2. Apply median imputation
      if (val.isNaN) {
        val = (imputerMedians[col] as num).toDouble();
      }

      // 3. Apply MinMaxScaler
      final minVal = (minmaxMin[col] as num).toDouble();
      final scaleVal = (minmaxScale[col] as num).toDouble();
      val = val * scaleVal + minVal;

      preprocessedVector[i] = val;
    }

    // 4. Feature Selection & Standardization (on selected features ONLY)
    final List<double> finalVector = List<double>.filled(selectedFeatures.length, 0.0);
    
    for (int i = 0; i < selectedFeatures.length; i++) {
      final col = selectedFeatures[i];
      final originalIndex = featureCols.indexOf(col);
      
      double val = (originalIndex != -1) ? preprocessedVector[originalIndex] : 0.0;
      
      // Apply StandardScaler
      final meanVal = (standardMean[col] as num).toDouble();
      final stdVal = (standardScale[col] as num).toDouble();
      
      val = (val - meanVal) / (stdVal > 1e-9 ? stdVal : 1.0);
      finalVector[i] = val;
    }

    // 5. Execute TFLite Inference via MLService
    try {
      return _mlService.runInference(finalVector);
    } catch (e) {
      return 0.5;
    }
  }

  /// Closes the interpreter session.
  void dispose() {
    _mlService.dispose();
  }
}
