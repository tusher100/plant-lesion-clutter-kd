import 'dart:io';
import 'dart:typed_data';

import 'package:path_provider/path_provider.dart';
import 'package:share_plus/share_plus.dart';

import 'segmenter.dart';

/// One saved check.
class LogEntry {
  LogEntry({required this.time, required this.photo, required this.severity, required this.note, required this.line});

  final DateTime time;
  final File photo;
  final double severity;
  final String note;
  final String line; // raw CSV line, used to delete the entry
}

/// Field log: every saved photo is copied into the app folder and a CSV row is appended. The CSV + photos are the
/// field-test data for the paper (label the lesions later and compare with the app's severity).
class FieldLog {
  FieldLog._(this.dir);

  final Directory dir;
  static const _header = 'timestamp,photo,model,severity_pct,severity_class,lesion_px,leaf_px,inference_ms,'
      'preprocess_ms,accelerator,note';

  static Future<FieldLog> open() async {
    final base = await getApplicationDocumentsDirectory();
    final d = Directory('${base.path}/field_test');
    await Directory('${d.path}/photos').create(recursive: true);
    final csv = File('${d.path}/field_log.csv');
    final lines = await csv.exists() ? await csv.readAsLines() : <String>[];
    if (lines.isEmpty || (lines.first != _header && lines.length <= 1)) {
      await csv.writeAsString('$_header\n'); // new log, or an empty log from an older app version
    }
    return FieldLog._(d);
  }

  File get csv => File('${dir.path}/field_log.csv');

  Future<List<LogEntry>> entries() async {
    final lines = await csv.readAsLines();
    final out = <LogEntry>[];
    for (final l in lines.skip(1)) {
      final c = l.split(',');
      if (c.length < 5) continue;
      final f = File('${dir.path}/photos/${c[1]}');
      if (!await f.exists()) continue;
      out.add(LogEntry(
        time: DateTime.tryParse(c[0]) ?? DateTime.now(),
        photo: f,
        severity: double.tryParse(c[3]) ?? 0,
        note: c.length > 10 ? c.sublist(10).join(',') : '',
        line: l,
      ));
    }
    return out.reversed.toList(); // newest first
  }

  Future<void> add(Uint8List photo, SegResult r, String model, String accel, String note) async {
    final ts = DateTime.now();
    final name = 'IMG_${ts.toIso8601String().replaceAll(':', '-').split('.').first}.jpg';
    await File('${dir.path}/photos/$name').writeAsBytes(photo);
    final clean = note.replaceAll(',', ';').replaceAll('\n', ' ');
    await csv.writeAsString(
        '${ts.toIso8601String()},$name,$model,${r.severityPct.toStringAsFixed(2)},${severityLevel(r.severityPct).name},'
        '${r.lesionPx},${r.leafPx},${r.inferenceMs.toStringAsFixed(1)},${r.preprocessMs.toStringAsFixed(1)},$accel,$clean\n',
        mode: FileMode.append);
  }

  Future<void> delete(LogEntry e) async {
    final lines = await csv.readAsLines();
    lines.remove(e.line);
    await csv.writeAsString('${lines.join('\n')}\n');
    if (await e.photo.exists()) await e.photo.delete();
  }

  /// Share the CSV and all photos (Google Drive, e-mail, ...).
  Future<void> export() async {
    final photos = Directory('${dir.path}/photos').listSync().whereType<File>().map((f) => XFile(f.path)).toList();
    await SharePlus.instance.share(ShareParams(files: [XFile(csv.path), ...photos], text: 'Leaf Check field log'));
  }
}
