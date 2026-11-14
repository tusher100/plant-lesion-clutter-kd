import 'field_log.dart';
import 'segmenter.dart';

/// Model and log are loaded once and shared by all screens.
class AppState {
  AppState._();

  static Segmenter? seg;
  static FieldLog? log;
  static String? error;
  static bool initialized = false;

  static Future<void> init() async {
    if (initialized) return;
    try {
      seg = await Segmenter.load();
    } catch (e) {
      error = 'modelMissing';
    }
    log ??= await FieldLog.open();
    initialized = true;
  }
}
