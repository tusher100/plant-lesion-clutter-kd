import 'dart:ffi';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:flutter/foundation.dart';
import 'package:image/image.dart' as img;
import 'package:tflite_flutter/tflite_flutter.dart';

/// Result of one photo: lesion mask at model resolution, leaf mask (proxy), severity and timing.
class SegResult {
  SegResult({
    required this.size,
    required this.lesion,
    required this.leaf,
    required this.inferenceMs,
    required this.preprocessMs,
  });

  final int size; // model resolution (e.g. 320)
  final Uint8List lesion; // size*size, 1 = lesion
  final Uint8List leaf; // size*size, 1 = leaf proxy OR lesion
  final double inferenceMs;
  final double preprocessMs;

  int get lesionPx => lesion.fold(0, (a, b) => a + b);
  int get leafPx => leaf.fold(0, (a, b) => a + b);

  /// Severity = lesion pixels / (leaf proxy U lesion) pixels, in percent (same definition as the paper's REV-4 check).
  double get severityPct => leafPx == 0 ? 0 : 100.0 * lesionPx / leafPx;
}

/// Severity classes used in the paper (0-5 %, 5-25 %, >25 % of leaf area). Illustrative, not a spray rule.
enum SeverityLevel { low, moderate, high }

SeverityLevel severityLevel(double pct) =>
    pct < 5 ? SeverityLevel.low : (pct < 25 ? SeverityLevel.moderate : SeverityLevel.high);

/// Input prepared in a background isolate (public for unit tests): normalised NHWC float pixels + the leaf proxy.
class Prepared {
  Prepared(this.rgb, this.leaf, this.ms);
  final Float32List rgb; // size*size*3, ImageNet-normalised, NHWC order
  final Uint8List leaf; // size*size, excess-green + Otsu (as in training)
  final double ms;
}

class PrepArgs {
  PrepArgs(this.bytes, this.size);
  final Uint8List bytes;
  final int size;
}

const _mean = [0.485, 0.456, 0.406];
const _std = [0.229, 0.224, 0.225];

/// Decode, fix EXIF orientation, resize the WHOLE photo to size x size (as in training, aspect not kept),
/// normalise, and compute the excess-green/Otsu leaf proxy on the resized image.
Prepared prepareImage(PrepArgs a) {
  final sw = Stopwatch()..start();
  final decoded = img.decodeImage(a.bytes);
  if (decoded == null) {
    throw Exception('Could not decode the image');
  }
  final oriented = img.bakeOrientation(decoded);
  final r = img.copyResize(oriented, width: a.size, height: a.size, interpolation: img.Interpolation.linear);
  final n = a.size * a.size;
  final rgb = Float32List(n * 3);
  final exg = Float32List(n);
  var mn = double.infinity, mx = -double.infinity;
  var i = 0;
  for (final p in r) {
    final rr = p.r / 255.0, gg = p.g / 255.0, bb = p.b / 255.0;
    rgb[i * 3] = (rr - _mean[0]) / _std[0];
    rgb[i * 3 + 1] = (gg - _mean[1]) / _std[1];
    rgb[i * 3 + 2] = (bb - _mean[2]) / _std[2];
    final e = 2 * gg - rr - bb;
    exg[i] = e;
    mn = math.min(mn, e);
    mx = math.max(mx, e);
    i++;
  }
  // min-max to 0..255 and Otsu threshold (cv2.THRESH_OTSU equivalent)
  final hist = List<int>.filled(256, 0);
  final q = Uint8List(n);
  final span = (mx - mn).abs() < 1e-9 ? 1.0 : (mx - mn);
  for (var k = 0; k < n; k++) {
    q[k] = ((exg[k] - mn) / span * 255).round().clamp(0, 255);
    hist[q[k]]++;
  }
  var sumAll = 0.0;
  for (var t = 0; t < 256; t++) {
    sumAll += t * hist[t];
  }
  var wB = 0.0, sumB = 0.0, best = -1.0, thr = 0;
  for (var t = 0; t < 256; t++) {
    wB += hist[t];
    if (wB == 0) continue;
    final wF = n - wB;
    if (wF == 0) break;
    sumB += t * hist[t];
    final mB = sumB / wB, mF = (sumAll - sumB) / wF;
    final between = wB * wF * (mB - mF) * (mB - mF);
    if (between > best) {
      best = between;
      thr = t;
    }
  }
  final leaf = Uint8List(n);
  for (var k = 0; k < n; k++) {
    leaf[k] = q[k] > thr ? 1 : 0;
  }
  return Prepared(rgb, leaf, sw.elapsedMicroseconds / 1000.0);
}

/// Wraps the TFLite student. Handles NHWC or NCHW layouts and float32 / int8 / uint8 input and output tensors,
/// so any of the exported variants (full INT8, dynamic range, FP16) can be dropped in as assets/model.tflite.
class Segmenter {
  Segmenter._(this._interp, this.size, this._nchw, this.modelName, this.accel);

  final Interpreter _interp;
  final int size;
  final bool _nchw;
  final String modelName;
  final String accel;

  static Future<Segmenter> load({String asset = 'assets/model.tflite', int threads = 4}) async {
    Interpreter interp;
    var accel = 'CPU reference';
    try {
      // XNNPACK with int8 (QS8) and uint8 (QU8) kernels enabled: without these flags an INT8 model falls back to
      // slow reference kernels (seconds per image instead of ~20 ms).
      final xo = XNNPackDelegateOptions(numThreads: threads);
      xo.base.ref.flags = 1 | 2; // TFLITE_XNNPACK_DELEGATE_FLAG_QS8 | TFLITE_XNNPACK_DELEGATE_FLAG_QU8
      final options = InterpreterOptions()
        ..threads = threads
        ..addDelegate(XNNPackDelegate(options: xo));
      interp = await Interpreter.fromAsset(asset, options: options);
      accel = 'XNNPACK x$threads';
    } catch (_) {
      interp = await Interpreter.fromAsset(asset, options: InterpreterOptions()..threads = threads);
    }
    interp.allocateTensors();
    final shape = interp.getInputTensor(0).shape; // [1, 320, 320, 3] (NHWC) or [1, 3, 320, 320] (NCHW)
    final nchw = shape.length == 4 && shape[1] == 3;
    final size = nchw ? shape[2] : shape[1];
    return Segmenter._(interp, size, nchw, asset.split('/').last, accel);
  }

  String describe() {
    final i = _interp.getInputTensor(0), o = _interp.getOutputTensor(0);
    return 'in ${i.shape} ${i.type} | out ${o.shape} ${o.type} | $accel';
  }

  Future<SegResult> run(Uint8List photoBytes) async {
    final prep = await compute(prepareImage, PrepArgs(photoBytes, size));
    final n = size * size;

    // ---- input
    final inT = _interp.getInputTensor(0);
    Float32List src = prep.rgb;
    if (_nchw) {
      final t = Float32List(n * 3);
      for (var k = 0; k < n; k++) {
        for (var c = 0; c < 3; c++) {
          t[c * n + k] = src[k * 3 + c];
        }
      }
      src = t;
    }
    Object input;
    if (inT.type == TensorType.float32) {
      input = src.buffer.asUint8List();
    } else {
      final sc = inT.params.scale, zp = inT.params.zeroPoint;
      if (inT.type == TensorType.int8) {
        final q = Int8List(src.length);
        for (var k = 0; k < src.length; k++) {
          q[k] = (src[k] / sc + zp).round().clamp(-128, 127);
        }
        input = q.buffer.asUint8List();
      } else {
        final q = Uint8List(src.length);
        for (var k = 0; k < src.length; k++) {
          q[k] = (src[k] / sc + zp).round().clamp(0, 255);
        }
        input = q;
      }
    }

    // ---- inference
    final outT = _interp.getOutputTensor(0);
    final outShape = outT.shape; // [1, H, W, 2] or [1, 2, H, W]
    final outNchw = outShape.length == 4 && outShape[1] == 2;
    final out = ByteData(outT.numBytes()).buffer;
    final sw = Stopwatch()..start();
    _interp.run(input, out);
    final ms = sw.elapsedMicroseconds / 1000.0;

    // ---- argmax over the 2 classes (quantised outputs share one scale, so argmax on raw values is exact)
    double v(int k) {
      switch (outT.type) {
        case TensorType.float32:
          return out.asFloat32List()[k];
        case TensorType.int8:
          return out.asInt8List()[k].toDouble();
        default:
          return out.asUint8List()[k].toDouble();
      }
    }

    final lesion = Uint8List(n);
    for (var k = 0; k < n; k++) {
      final bg = outNchw ? v(k) : v(k * 2);
      final fg = outNchw ? v(n + k) : v(k * 2 + 1);
      lesion[k] = fg > bg ? 1 : 0;
    }
    final leaf = Uint8List(n);
    for (var k = 0; k < n; k++) {
      leaf[k] = (prep.leaf[k] == 1 || lesion[k] == 1) ? 1 : 0;
    }
    return SegResult(size: size, lesion: lesion, leaf: leaf, inferenceMs: ms, preprocessMs: prep.ms);
  }

  void close() => _interp.close();
}

/// Semi-transparent red overlay (lesion) as PNG bytes at model resolution; the UI stretches it over the photo,
/// which inverts the training-time resize exactly.
Uint8List overlayPng(SegResult r) {
  final im = img.Image(width: r.size, height: r.size, numChannels: 4);
  for (var y = 0; y < r.size; y++) {
    for (var x = 0; x < r.size; x++) {
      final k = y * r.size + x;
      if (r.lesion[k] == 1) {
        im.setPixelRgba(x, y, 230, 30, 30, 140);
      } else {
        im.setPixelRgba(x, y, 0, 0, 0, 0);
      }
    }
  }
  return Uint8List.fromList(img.encodePng(im));
}
