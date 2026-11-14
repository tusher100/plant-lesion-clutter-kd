import 'dart:typed_data';

import 'package:flutter/material.dart';

import 'app_state.dart';
import 'l10n.dart';
import 'segmenter.dart';
import 'widgets.dart';

class ResultScreen extends StatefulWidget {
  const ResultScreen({super.key, required this.photo, this.preset});

  final Uint8List photo;
  final SegResult? preset; // used only by the UI preview test

  @override
  State<ResultScreen> createState() => _ResultScreenState();
}

class _ResultScreenState extends State<ResultScreen> {
  SegResult? _r;
  Uint8List? _overlay;
  double _aspect = 1;
  bool _showArea = true;
  bool _saved = false;
  bool _failed = false;
  final _note = TextEditingController();

  @override
  void initState() {
    super.initState();
    _analyse();
  }

  @override
  void dispose() {
    _note.dispose();
    super.dispose();
  }

  Future<void> _analyse() async {
    try {
      final img = await decodeImageFromList(widget.photo);
      final r = widget.preset ?? await AppState.seg!.run(widget.photo);
      if (!mounted) return;
      setState(() {
        _aspect = img.width / img.height;
        _r = r;
        _overlay = overlayPng(r);
      });
    } catch (_) {
      if (mounted) setState(() => _failed = true);
    }
  }

  Future<void> _save() async {
    final s = AppState.seg!;
    await AppState.log!.add(widget.photo, _r!, s.modelName, s.accel, _note.text);
    if (!mounted) return;
    FocusScope.of(context).unfocus();
    setState(() => _saved = true);
    ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text(tr('saved'))));
  }

  @override
  Widget build(BuildContext context) {
    final tt = Theme.of(context).textTheme;
    final cs = Theme.of(context).colorScheme;
    final r = _r;
    return Scaffold(
      appBar: AppBar(title: Text(tr('result'))),
      body: ListView(
        padding: const EdgeInsets.fromLTRB(20, 4, 20, 24),
        children: [
          // ---- photo with diseased area
          AspectRatio(
            aspectRatio: _aspect,
            child: ClipRRect(
              borderRadius: BorderRadius.circular(20),
              child: Stack(fit: StackFit.expand, children: [
                Image.memory(widget.photo, fit: BoxFit.fill, gaplessPlayback: true),
                if (_overlay != null)
                  AnimatedOpacity(
                    opacity: _showArea ? 1 : 0,
                    duration: const Duration(milliseconds: 250),
                    child: Image.memory(_overlay!, fit: BoxFit.fill, gaplessPlayback: true),
                  ),
                if (r == null && !_failed)
                  Container(
                    color: Colors.black38,
                    child: Center(
                      child: Column(mainAxisSize: MainAxisSize.min, children: [
                        const CircularProgressIndicator(color: Colors.white),
                        const SizedBox(height: 12),
                        Text(tr('analysing'), style: tt.titleMedium?.copyWith(color: Colors.white)),
                      ]),
                    ),
                  ),
              ]),
            ),
          ),
          if (_failed) ...[
            const SizedBox(height: 24),
            Text(tr('failed'), style: tt.titleMedium, textAlign: TextAlign.center),
            const SizedBox(height: 16),
            BigButton(icon: Icons.refresh_rounded, label: tr('again'), onPressed: () => Navigator.pop(context)),
          ],
          if (r != null) ...[
            SwitchListTile(
              contentPadding: EdgeInsets.zero,
              value: _showArea,
              onChanged: (v) => setState(() => _showArea = v),
              title: Text(tr('showArea'), style: tt.titleMedium),
              subtitle: Text(tr('legend')),
            ),
            const SizedBox(height: 4),
            // ---- result card
            Card(
              color: levelColor(severityLevel(r.severityPct)).withValues(alpha: 0.07),
              elevation: 0,
              shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(24)),
              child: Padding(
                padding: const EdgeInsets.fromLTRB(20, 20, 20, 20),
                child: Column(children: [
                  SeverityGauge(percent: r.severityPct),
                  Text(tr('diseased'), style: tt.titleMedium, textAlign: TextAlign.center),
                  const SizedBox(height: 14),
                  LevelChip(level: severityLevel(r.severityPct), big: true),
                  const SizedBox(height: 14),
                  Text(levelMessage(severityLevel(r.severityPct)), style: tt.bodyLarge, textAlign: TextAlign.center),
                ]),
              ),
            ),
            const SizedBox(height: 16),
            // ---- note + save
            TextField(
              controller: _note,
              enabled: !_saved,
              textInputAction: TextInputAction.done,
              decoration: InputDecoration(
                labelText: tr('note'),
                prefixIcon: const Icon(Icons.edit_note_rounded),
                border: OutlineInputBorder(borderRadius: BorderRadius.circular(16)),
              ),
            ),
            const SizedBox(height: 12),
            BigButton(
              icon: _saved ? Icons.check_rounded : Icons.bookmark_add_rounded,
              label: _saved ? tr('saved') : tr('save'),
              onPressed: _saved ? null : _save,
            ),
            const SizedBox(height: 12),
            BigButton(
              icon: Icons.photo_camera_rounded,
              label: tr('again'),
              filled: false,
              onPressed: () => Navigator.pop(context),
            ),
            const SizedBox(height: 20),
            Row(crossAxisAlignment: CrossAxisAlignment.start, children: [
              Icon(Icons.info_outline_rounded, size: 20, color: cs.onSurfaceVariant),
              const SizedBox(width: 8),
              Expanded(child: Text(tr('disclaimer'), style: tt.bodySmall?.copyWith(color: cs.onSurfaceVariant))),
            ]),
            // ---- technical details, hidden by default (for researchers)
            Theme(
              data: Theme.of(context).copyWith(dividerColor: Colors.transparent),
              child: ExpansionTile(
                tilePadding: EdgeInsets.zero,
                title: Text(tr('details'), style: tt.bodyMedium?.copyWith(color: cs.onSurfaceVariant)),
                children: [
                  _kv(tr('speed'), '${r.inferenceMs.toStringAsFixed(0)} ms  (+${r.preprocessMs.toStringAsFixed(0)} ms)'),
                  _kv('Severity', '${r.severityPct.toStringAsFixed(2)} %  (${r.lesionPx} / ${r.leafPx} px)'),
                  _kv('Model', AppState.seg?.modelName ?? '-'),
                  _kv('Runtime', AppState.seg?.accel ?? '-'),
                  _kv('Tensors', AppState.seg?.describe() ?? '-'),
                ],
              ),
            ),
          ],
        ],
      ),
    );
  }

  Widget _kv(String k, String v) => Padding(
        padding: const EdgeInsets.symmetric(vertical: 3),
        child: Row(crossAxisAlignment: CrossAxisAlignment.start, children: [
          SizedBox(width: 110, child: Text(k, style: const TextStyle(fontWeight: FontWeight.w600))),
          Expanded(child: Text(v)),
        ]),
      );
}
