import 'package:flutter/material.dart';

import 'home_screen.dart';
import 'l10n.dart';

void main() => runApp(const LeafCheckApp());

class LeafCheckApp extends StatelessWidget {
  const LeafCheckApp({super.key});

  @override
  Widget build(BuildContext context) {
    final base = ThemeData(colorSchemeSeed: const Color(0xFF2E7D32), useMaterial3: true);
    return ValueListenableBuilder<Lang>(
      valueListenable: appLang,
      builder: (context, lang, _) => MaterialApp(
        title: tr('app'),
        debugShowCheckedModeBanner: false,
        theme: base.copyWith(cardTheme: const CardThemeData(elevation: 0, margin: EdgeInsets.zero)),
        // slightly larger text (readable outdoors and for older users), on top of the phone's own text-size setting
        builder: (context, child) {
          final mq = MediaQuery.of(context);
          return MediaQuery(
            data: mq.copyWith(textScaler: TextScaler.linear(mq.textScaler.scale(14) / 14 * 1.08)),
            child: child!,
          );
        },
        home: const HomeScreen(),
      ),
    );
  }
}
