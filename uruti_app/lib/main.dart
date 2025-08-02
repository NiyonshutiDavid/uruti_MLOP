import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:flutter_spinkit/flutter_spinkit.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:flutter_sound/flutter_sound.dart';
import 'dart:io';

void main() => runApp(MissionClassifierApp());

class MissionClassifierApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      title: 'Mission Classifier',
      theme: ThemeData(
        scaffoldBackgroundColor: Colors.black,
        textTheme: GoogleFonts.interTightTextTheme(
          Theme.of(context).textTheme.apply(
                bodyColor: Colors.white,
                displayColor: Colors.white,
              ),
        ).apply(
          bodyColor: Colors.white,
          displayColor: Colors.white,
        ),
      ),
      home: MissionInputPage(),
    );
  }
}

class MissionInputPage extends StatefulWidget {
  @override
  _MissionInputPageState createState() => _MissionInputPageState();
}

class _MissionInputPageState extends State<MissionInputPage> with SingleTickerProviderStateMixin {
  final FlutterSoundRecorder _recorder = FlutterSoundRecorder();
  bool _isRecording = false;
  String? _audioPath;
  late AnimationController _controller;
  late Animation<double> _animation;

  final TextEditingController _missionController = TextEditingController();
  final TextEditingController _visionController = TextEditingController();
  final TextEditingController _problemController = TextEditingController();
  final TextEditingController _solutionController = TextEditingController();

  @override
  void initState() {
    super.initState();
    _controller = AnimationController(
      duration: const Duration(seconds: 2),
      vsync: this,
    )..repeat(reverse: true);
    _animation = CurvedAnimation(
      parent: _controller,
      curve: Curves.easeInOut,
    );
    _initRecorder();
  }

  Future<void> _initRecorder() async {
    await Permission.microphone.request();
    await _recorder.openRecorder();
  }

  Future<void> _startRecording() async {
    Directory tempDir = Directory.systemTemp;
    _audioPath = '${tempDir.path}/mission_input.wav';
    await _recorder.startRecorder(toFile: _audioPath);
    setState(() => _isRecording = true);
  }

  Future<void> _stopRecording() async {
    await _recorder.stopRecorder();
    setState(() => _isRecording = false);
    Navigator.push(
      context,
      MaterialPageRoute(
        builder: (context) => ResultPage(audioPath: _audioPath!),
      ),
    );
  }

  @override
  void dispose() {
    _recorder.closeRecorder();
    _controller.dispose();
    _missionController.dispose();
    _visionController.dispose();
    _problemController.dispose();
    _solutionController.dispose();
    super.dispose();
  }

  Widget _buildInputField(String label, TextEditingController controller) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 6.0),
      child: TextField(
        controller: controller,
        style: TextStyle(letterSpacing: -0.6, color: Colors.white),
        decoration: InputDecoration(
          labelText: label,
          labelStyle: TextStyle(color: Colors.green.shade300),
          filled: true,
          fillColor: Colors.grey.shade900,
          border: OutlineInputBorder(borderRadius: BorderRadius.circular(12)),
        ),
        maxLines: null,
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Mission Classifier'),
        backgroundColor: Colors.green.shade700,
        centerTitle: true,
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            _buildInputField('Mission', _missionController),
            _buildInputField('Vision', _visionController),
            _buildInputField('Problem Statement', _problemController),
            _buildInputField('Proposed Solution', _solutionController),
            const SizedBox(height: 20),
            Center(
              child: Column(
                children: [
                  if (_isRecording)
                    ScaleTransition(
                      scale: _animation,
                      child: Icon(Icons.mic, color: Colors.green, size: 64),
                    )
                  else
                    Icon(Icons.mic_none, color: Colors.white, size: 64),
                  const SizedBox(height: 10),
                  ElevatedButton(
                    onPressed: _isRecording ? _stopRecording : _startRecording,
                    style: ElevatedButton.styleFrom(
                      backgroundColor: Colors.green.shade700,
                      padding: const EdgeInsets.symmetric(horizontal: 30, vertical: 16),
                      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
                    ),
                    child: Text(
                      _isRecording ? 'Stop & Analyze' : 'Start Recording',
                      style: const TextStyle(fontSize: 16, letterSpacing: -0.6),
                    ),
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class ResultPage extends StatelessWidget {
  final String audioPath;

  const ResultPage({super.key, required this.audioPath});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.black,
      appBar: AppBar(
        title: const Text('Results Summary'),
        backgroundColor: Colors.green.shade700,
      ),
      body: Center(
        child: Padding(
          padding: const EdgeInsets.all(24.0),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              const SpinKitThreeBounce(color: Colors.green, size: 50),
              const SizedBox(height: 20),
              const Text(
                'Analyzing your input...\nThis may take a few seconds.',
                textAlign: TextAlign.center,
                style: TextStyle(fontSize: 18, letterSpacing: -0.6),
              ),
              const SizedBox(height: 40),
              const Text(
                '(Model output and classification results will be shown here after integration)',
                textAlign: TextAlign.center,
                style: TextStyle(fontSize: 14, color: Colors.grey),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
