import 'package:flutter/foundation.dart';

/// App language (English / Bahasa Melayu). Switch from the language button in the top bar.
enum Lang { en, ms }

final ValueNotifier<Lang> appLang = ValueNotifier(Lang.en);

const Map<String, Map<Lang, String>> _t = {
  'app': {Lang.en: 'Leaf Check', Lang.ms: 'Semak Daun'},
  'tagline': {
    Lang.en: 'See how much of a leaf is diseased. Works without internet.',
    Lang.ms: 'Lihat berapa banyak daun yang berpenyakit. Berfungsi tanpa internet.',
  },
  'scan': {Lang.en: 'Take a photo', Lang.ms: 'Ambil gambar'},
  'pick': {Lang.en: 'Choose from gallery', Lang.ms: 'Pilih dari galeri'},
  'tipsTitle': {Lang.en: 'For a good result', Lang.ms: 'Untuk hasil yang baik'},
  'tip1': {Lang.en: 'One leaf fills most of the photo', Lang.ms: 'Satu daun memenuhi sebahagian besar gambar'},
  'tip2': {Lang.en: 'Daylight, no strong shadow', Lang.ms: 'Cahaya siang, tiada bayang yang kuat'},
  'tip3': {Lang.en: 'Hold the phone still', Lang.ms: 'Pegang telefon dengan stabil'},
  'recent': {Lang.en: 'Recent checks', Lang.ms: 'Semakan terkini'},
  'seeAll': {Lang.en: 'See all', Lang.ms: 'Lihat semua'},
  'noHistory': {
    Lang.en: 'Your saved checks will appear here.',
    Lang.ms: 'Semakan yang disimpan akan dipaparkan di sini.',
  },
  'loading': {Lang.en: 'Preparing...', Lang.ms: 'Sedang bersedia...'},
  'analysing': {Lang.en: 'Checking the leaf...', Lang.ms: 'Sedang menyemak daun...'},
  'modelMissing': {
    Lang.en: 'The model file is missing. Add assets/model.tflite and restart the app.',
    Lang.ms: 'Fail model tiada. Tambah assets/model.tflite dan mulakan semula aplikasi.',
  },
  'failed': {Lang.en: 'Could not check this photo. Please try another one.', Lang.ms: 'Gambar ini tidak dapat disemak. Sila cuba gambar lain.'},
  'result': {Lang.en: 'Result', Lang.ms: 'Keputusan'},
  'diseased': {Lang.en: 'of the leaf looks diseased', Lang.ms: 'daripada daun kelihatan berpenyakit'},
  'low': {Lang.en: 'Low', Lang.ms: 'Rendah'},
  'moderate': {Lang.en: 'Moderate', Lang.ms: 'Sederhana'},
  'high': {Lang.en: 'High', Lang.ms: 'Tinggi'},
  'lowMsg': {
    Lang.en: 'Only a small area is affected. Keep watching this plant.',
    Lang.ms: 'Hanya kawasan kecil terjejas. Terus perhatikan pokok ini.',
  },
  'moderateMsg': {
    Lang.en: 'A clear part of the leaf is affected. Check nearby plants too.',
    Lang.ms: 'Sebahagian jelas daun terjejas. Periksa juga pokok berhampiran.',
  },
  'highMsg': {
    Lang.en: 'A large part of the leaf is affected. Ask an agriculture officer for advice.',
    Lang.ms: 'Sebahagian besar daun terjejas. Dapatkan nasihat pegawai pertanian.',
  },
  'showArea': {Lang.en: 'Show diseased area', Lang.ms: 'Tunjuk kawasan berpenyakit'},
  'legend': {Lang.en: 'Red = diseased area found by the app', Lang.ms: 'Merah = kawasan berpenyakit yang dikesan'},
  'note': {Lang.en: 'Add a note (crop, place) - optional', Lang.ms: 'Tambah catatan (tanaman, tempat) - pilihan'},
  'save': {Lang.en: 'Save', Lang.ms: 'Simpan'},
  'saved': {Lang.en: 'Saved', Lang.ms: 'Disimpan'},
  'again': {Lang.en: 'Check another leaf', Lang.ms: 'Semak daun lain'},
  'disclaimer': {
    Lang.en: 'This is an estimate from the photo. It does not name the disease and is not a spraying instruction.',
    Lang.ms: 'Ini anggaran daripada gambar. Ia tidak menamakan penyakit dan bukan arahan semburan.',
  },
  'details': {Lang.en: 'Technical details', Lang.ms: 'Butiran teknikal'},
  'speed': {Lang.en: 'Analysis time', Lang.ms: 'Masa analisis'},
  'history': {Lang.en: 'Saved checks', Lang.ms: 'Semakan disimpan'},
  'export': {Lang.en: 'Share all', Lang.ms: 'Kongsi semua'},
  'delete': {Lang.en: 'Delete', Lang.ms: 'Padam'},
  'deleteAsk': {Lang.en: 'Delete this check?', Lang.ms: 'Padam semakan ini?'},
  'cancel': {Lang.en: 'Cancel', Lang.ms: 'Batal'},
  'language': {Lang.en: 'Bahasa Melayu', Lang.ms: 'English'},
};

/// Translate a key into the current language.
String tr(String key) => _t[key]?[appLang.value] ?? key;
