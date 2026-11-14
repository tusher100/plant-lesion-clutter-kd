import 'package:flutter/material.dart';

import 'app_state.dart';
import 'field_log.dart';
import 'l10n.dart';
import 'segmenter.dart';
import 'widgets.dart';

class HistoryScreen extends StatefulWidget {
  const HistoryScreen({super.key});

  @override
  State<HistoryScreen> createState() => _HistoryScreenState();
}

class _HistoryScreenState extends State<HistoryScreen> {
  List<LogEntry> _items = [];

  @override
  void initState() {
    super.initState();
    _load();
  }

  Future<void> _load() async {
    final e = await AppState.log!.entries();
    if (mounted) setState(() => _items = e);
  }

  Future<void> _delete(LogEntry e) async {
    final ok = await showDialog<bool>(
      context: context,
      builder: (c) => AlertDialog(
        title: Text(tr('deleteAsk')),
        actions: [
          TextButton(onPressed: () => Navigator.pop(c, false), child: Text(tr('cancel'))),
          FilledButton(onPressed: () => Navigator.pop(c, true), child: Text(tr('delete'))),
        ],
      ),
    );
    if (ok == true) {
      await AppState.log!.delete(e);
      await _load();
    }
  }

  String _date(DateTime t) =>
      '${t.day.toString().padLeft(2, '0')}/${t.month.toString().padLeft(2, '0')}/${t.year}  '
      '${t.hour.toString().padLeft(2, '0')}:${t.minute.toString().padLeft(2, '0')}';

  @override
  Widget build(BuildContext context) {
    final tt = Theme.of(context).textTheme;
    return Scaffold(
      appBar: AppBar(
        title: Text(tr('history')),
        actions: [
          if (_items.isNotEmpty)
            TextButton.icon(
              onPressed: () => AppState.log!.export(),
              icon: const Icon(Icons.ios_share_rounded),
              label: Text(tr('export')),
            ),
        ],
      ),
      body: _items.isEmpty
          ? Center(child: Text(tr('noHistory'), style: tt.bodyLarge))
          : ListView.separated(
              padding: const EdgeInsets.all(16),
              itemCount: _items.length,
              separatorBuilder: (_, __) => const SizedBox(height: 10),
              itemBuilder: (_, i) {
                final e = _items[i];
                final level = severityLevel(e.severity);
                return Card(
                  elevation: 0,
                  color: Theme.of(context).colorScheme.surfaceContainerHigh,
                  clipBehavior: Clip.antiAlias,
                  child: Row(children: [
                    Image.file(e.photo, width: 96, height: 96, fit: BoxFit.cover, cacheWidth: 240),
                    const SizedBox(width: 14),
                    Expanded(
                      child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                        Row(children: [
                          Text('${e.severity.toStringAsFixed(0)}%',
                              style: tt.titleLarge?.copyWith(fontWeight: FontWeight.w700)),
                          const SizedBox(width: 10),
                          LevelChip(level: level),
                        ]),
                        const SizedBox(height: 4),
                        Text(_date(e.time), style: tt.bodyMedium),
                        if (e.note.isNotEmpty)
                          Text(e.note, maxLines: 1, overflow: TextOverflow.ellipsis, style: tt.bodySmall),
                      ]),
                    ),
                    IconButton(
                      tooltip: tr('delete'),
                      icon: const Icon(Icons.delete_outline_rounded),
                      onPressed: () => _delete(e),
                    ),
                  ]),
                );
              },
            ),
    );
  }
}
