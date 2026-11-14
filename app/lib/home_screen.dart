import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';

import 'app_state.dart';
import 'field_log.dart';
import 'history_screen.dart';
import 'l10n.dart';
import 'result_screen.dart';
import 'segmenter.dart';
import 'widgets.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  bool _ready = false;
  List<LogEntry> _recent = [];

  void _onLang() => setState(() {});

  @override
  void dispose() {
    appLang.removeListener(_onLang);
    super.dispose();
  }

  @override
  void initState() {
    super.initState();
    appLang.addListener(_onLang);
    AppState.init().then((_) async {
      await _refresh();
      if (mounted) setState(() => _ready = true);
    });
  }

  Future<void> _refresh() async {
    final e = await AppState.log?.entries() ?? <LogEntry>[];
    if (mounted) setState(() => _recent = e);
  }

  Future<void> _start(ImageSource src) async {
    final f = await ImagePicker().pickImage(source: src, maxWidth: 1600, maxHeight: 1600, imageQuality: 92);
    if (f == null || !mounted) return;
    final bytes = await f.readAsBytes();
    if (!mounted) return;
    await Navigator.of(context).push(MaterialPageRoute(builder: (_) => ResultScreen(photo: bytes)));
    await _refresh();
  }

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    final tt = Theme.of(context).textTheme;
    return Scaffold(
      appBar: AppBar(
        title: Text(tr('app'), style: const TextStyle(fontWeight: FontWeight.w700)),
        actions: [
          TextButton.icon(
            onPressed: () => appLang.value = appLang.value == Lang.en ? Lang.ms : Lang.en,
            icon: const Icon(Icons.translate),
            label: Text(tr('language')),
          ),
        ],
      ),
      body: !_ready
          ? Center(
              child: Column(mainAxisSize: MainAxisSize.min, children: [
                const CircularProgressIndicator(),
                const SizedBox(height: 16),
                Text(tr('loading'), style: tt.titleMedium),
              ]),
            )
          : ListView(
              padding: const EdgeInsets.fromLTRB(20, 8, 20, 24),
              children: [
                // ---- hero
                Container(
                  padding: const EdgeInsets.all(20),
                  decoration: BoxDecoration(
                    color: cs.primaryContainer,
                    borderRadius: BorderRadius.circular(24),
                  ),
                  child: Row(children: [
                    Container(
                      width: 64,
                      height: 64,
                      decoration: BoxDecoration(color: cs.primary, shape: BoxShape.circle),
                      child: Icon(Icons.eco_rounded, color: cs.onPrimary, size: 36),
                    ),
                    const SizedBox(width: 16),
                    Expanded(
                      child: Text(tr('tagline'), style: tt.titleMedium?.copyWith(color: cs.onPrimaryContainer)),
                    ),
                  ]),
                ),
                const SizedBox(height: 20),
                if (AppState.error != null)
                  _Banner(text: tr(AppState.error!))
                else ...[
                  BigButton(icon: Icons.photo_camera_rounded, label: tr('scan'), onPressed: () => _start(ImageSource.camera)),
                  const SizedBox(height: 12),
                  BigButton(
                    icon: Icons.photo_library_rounded,
                    label: tr('pick'),
                    filled: false,
                    onPressed: () => _start(ImageSource.gallery),
                  ),
                ],
                const SizedBox(height: 24),
                // ---- tips
                Text(tr('tipsTitle'), style: tt.titleLarge?.copyWith(fontWeight: FontWeight.w700)),
                const SizedBox(height: 10),
                _Tip(icon: Icons.crop_free_rounded, text: tr('tip1')),
                _Tip(icon: Icons.wb_sunny_rounded, text: tr('tip2')),
                _Tip(icon: Icons.back_hand_rounded, text: tr('tip3')),
                const SizedBox(height: 20),
                // ---- recent
                Row(children: [
                  Expanded(child: Text(tr('recent'), style: tt.titleLarge?.copyWith(fontWeight: FontWeight.w700))),
                  if (_recent.isNotEmpty)
                    TextButton(
                      onPressed: () async {
                        await Navigator.of(context).push(MaterialPageRoute(builder: (_) => const HistoryScreen()));
                        await _refresh();
                      },
                      child: Text(tr('seeAll')),
                    ),
                ]),
                const SizedBox(height: 8),
                if (_recent.isEmpty)
                  Padding(
                    padding: const EdgeInsets.symmetric(vertical: 12),
                    child: Text(tr('noHistory'), style: tt.bodyLarge?.copyWith(color: cs.onSurfaceVariant)),
                  )
                else
                  SizedBox(
                    height: 150,
                    child: ListView.separated(
                      scrollDirection: Axis.horizontal,
                      itemCount: _recent.length.clamp(0, 10),
                      separatorBuilder: (_, __) => const SizedBox(width: 12),
                      itemBuilder: (_, i) => _RecentCard(e: _recent[i]),
                    ),
                  ),
              ],
            ),
    );
  }
}

class _Tip extends StatelessWidget {
  const _Tip({required this.icon, required this.text});

  final IconData icon;
  final String text;

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 6),
      child: Row(children: [
        CircleAvatar(radius: 20, backgroundColor: cs.secondaryContainer, child: Icon(icon, color: cs.onSecondaryContainer)),
        const SizedBox(width: 14),
        Expanded(child: Text(text, style: Theme.of(context).textTheme.bodyLarge)),
      ]),
    );
  }
}

class _RecentCard extends StatelessWidget {
  const _RecentCard({required this.e});

  final LogEntry e;

  @override
  Widget build(BuildContext context) {
    final level = severityLevel(e.severity);
    return SizedBox(
      width: 130,
      child: Card(
        elevation: 0,
        clipBehavior: Clip.antiAlias,
        color: Theme.of(context).colorScheme.surfaceContainerHigh,
        child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
          Expanded(child: Image.file(e.photo, fit: BoxFit.cover, width: double.infinity, cacheWidth: 300)),
          Padding(
            padding: const EdgeInsets.all(8),
            child: Row(children: [
              Text('${e.severity.toStringAsFixed(0)}%', style: const TextStyle(fontWeight: FontWeight.w700, fontSize: 16)),
              const Spacer(),
              Icon(levelIcon(level), color: levelColor(level), size: 20),
            ]),
          ),
        ]),
      ),
    );
  }
}

class _Banner extends StatelessWidget {
  const _Banner({required this.text});

  final String text;

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(color: cs.errorContainer, borderRadius: BorderRadius.circular(16)),
      child: Row(children: [
        Icon(Icons.error_outline, color: cs.onErrorContainer),
        const SizedBox(width: 12),
        Expanded(child: Text(text, style: TextStyle(color: cs.onErrorContainer))),
      ]),
    );
  }
}
