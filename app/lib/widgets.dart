import 'dart:math' as math;

import 'package:flutter/material.dart';

import 'l10n.dart';
import 'segmenter.dart';

/// Colours used for the three severity levels; always shown together with a word, never colour alone.
Color levelColor(SeverityLevel l) => switch (l) {
      SeverityLevel.low => const Color(0xFF2E7D32),
      SeverityLevel.moderate => const Color(0xFFE08A00),
      SeverityLevel.high => const Color(0xFFC62828),
    };

IconData levelIcon(SeverityLevel l) => switch (l) {
      SeverityLevel.low => Icons.check_circle_rounded,
      SeverityLevel.moderate => Icons.error_rounded,
      SeverityLevel.high => Icons.warning_rounded,
    };

String levelName(SeverityLevel l) => tr(l.name);

String levelMessage(SeverityLevel l) => tr('${l.name}Msg');

/// Half-ring gauge: green / amber / red zones (0-5, 5-25, 25-100 %) with a needle at the severity value.
class SeverityGauge extends StatelessWidget {
  const SeverityGauge({super.key, required this.percent});

  final double percent;

  @override
  Widget build(BuildContext context) {
    final level = severityLevel(percent);
    return Semantics(
      label: '${percent.toStringAsFixed(0)} percent, ${levelName(level)}',
      child: SizedBox(
        width: 240,
        child: TweenAnimationBuilder<double>(
          tween: Tween(begin: 0, end: percent.clamp(0, 100)),
          duration: const Duration(milliseconds: 900),
          curve: Curves.easeOutCubic,
          builder: (context, v, _) => Column(mainAxisSize: MainAxisSize.min, children: [
            SizedBox(
              width: 240,
              height: 120,
              child: CustomPaint(painter: _GaugePainter(v, Theme.of(context).colorScheme.onSurface)),
            ),
            Text('${v.toStringAsFixed(0)}%',
                style: Theme.of(context).textTheme.displaySmall?.copyWith(fontWeight: FontWeight.w800)),
          ]),
        ),
      ),
    );
  }
}

class _GaugePainter extends CustomPainter {
  _GaugePainter(this.value, this.ink);

  final double value;
  final Color ink;

  // The gauge is not linear: the low and moderate bands get more room, because most leaves fall there.
  double _pos(double p) => p <= 5 ? p / 5 * 0.25 : (p <= 25 ? 0.25 + (p - 5) / 20 * 0.4 : 0.65 + (p - 25) / 75 * 0.35);

  @override
  void paint(Canvas canvas, Size size) {
    final c = Offset(size.width / 2, size.height - 10);
    final r = math.min(size.width / 2, size.height) - 14;
    final rect = Rect.fromCircle(center: c, radius: r);
    final bands = [
      (0.0, 0.25, levelColor(SeverityLevel.low)),
      (0.25, 0.65, levelColor(SeverityLevel.moderate)),
      (0.65, 1.0, levelColor(SeverityLevel.high)),
    ];
    for (final (a, b, col) in bands) {
      canvas.drawArc(rect, math.pi + a * math.pi + 0.02, (b - a) * math.pi - 0.04, false,
          Paint()
            ..color = col.withValues(alpha: 0.9)
            ..style = PaintingStyle.stroke
            ..strokeWidth = 18
            ..strokeCap = StrokeCap.butt);
    }
    final ang = math.pi + _pos(value) * math.pi;
    final tip = c + Offset(math.cos(ang), math.sin(ang)) * (r - 4);
    canvas.drawLine(c, tip, Paint()
      ..color = ink
      ..strokeWidth = 4
      ..strokeCap = StrokeCap.round);
    canvas.drawCircle(c, 8, Paint()..color = ink);
  }

  @override
  bool shouldRepaint(_GaugePainter old) => old.value != value;
}

/// Small coloured pill: icon + level word.
class LevelChip extends StatelessWidget {
  const LevelChip({super.key, required this.level, this.big = false});

  final SeverityLevel level;
  final bool big;

  @override
  Widget build(BuildContext context) {
    final col = levelColor(level);
    return Container(
      padding: EdgeInsets.symmetric(horizontal: big ? 16 : 10, vertical: big ? 8 : 4),
      decoration: BoxDecoration(color: col.withValues(alpha: 0.12), borderRadius: BorderRadius.circular(40)),
      child: Row(mainAxisSize: MainAxisSize.min, children: [
        Icon(levelIcon(level), color: col, size: big ? 24 : 16),
        const SizedBox(width: 6),
        Text(levelName(level),
            style: TextStyle(color: col, fontWeight: FontWeight.w700, fontSize: big ? 20 : 13)),
      ]),
    );
  }
}

/// Big, full-width action button with icon (easy to hit with a thumb, readable outdoors).
class BigButton extends StatelessWidget {
  const BigButton({super.key, required this.icon, required this.label, required this.onPressed, this.filled = true});

  final IconData icon;
  final String label;
  final VoidCallback? onPressed;
  final bool filled;

  @override
  Widget build(BuildContext context) {
    final style = ButtonStyle(
      minimumSize: const WidgetStatePropertyAll(Size.fromHeight(64)),
      shape: WidgetStatePropertyAll(RoundedRectangleBorder(borderRadius: BorderRadius.circular(18))),
      textStyle: WidgetStatePropertyAll(
          Theme.of(context).textTheme.titleMedium?.copyWith(fontSize: 18, fontWeight: FontWeight.w600)),
    );
    final child = Row(mainAxisAlignment: MainAxisAlignment.center, children: [
      Icon(icon, size: 28),
      const SizedBox(width: 12),
      Flexible(child: Text(label, overflow: TextOverflow.ellipsis)),
    ]);
    return filled
        ? FilledButton(style: style, onPressed: onPressed, child: child)
        : OutlinedButton(style: style, onPressed: onPressed, child: child);
  }
}
