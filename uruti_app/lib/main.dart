import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:flutter_spinkit/flutter_spinkit.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:flutter_sound/flutter_sound.dart';
import 'package:http/http.dart' as http;
import 'package:http_parser/http_parser.dart';
import 'package:file_picker/file_picker.dart';
import 'package:path/path.dart' as path;
import 'dart:io';
import 'dart:convert';
import 'dart:typed_data';

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
      home: SplashScreen(),
    );
  }
}

// Analysis History Model
class AnalysisHistory {
  final String id;
  final String inputText;
  final String? audioPath;
  final Map<String, dynamic> result;
  final DateTime timestamp;
  final String inputType; // 'text', 'audio', 'upload'

  AnalysisHistory({
    required this.id,
    required this.inputText,
    this.audioPath,
    required this.result,
    required this.timestamp,
    required this.inputType,
  });

  Map<String, dynamic> toJson() {
    return {
      'id': id,
      'inputText': inputText,
      'audioPath': audioPath,
      'result': result,
      'timestamp': timestamp.toIso8601String(),
      'inputType': inputType,
    };
  }

  factory AnalysisHistory.fromJson(Map<String, dynamic> json) {
    return AnalysisHistory(
      id: json['id'],
      inputText: json['inputText'],
      audioPath: json['audioPath'],
      result: Map<String, dynamic>.from(json['result']),
      timestamp: DateTime.parse(json['timestamp']),
      inputType: json['inputType'],
    );
  }
}

// History Manager
class HistoryManager {
  static List<AnalysisHistory> _history = [];

  static List<AnalysisHistory> get history => _history;

  static void addHistory(AnalysisHistory item) {
    _history.insert(0, item);
    // Keep only last 50 items
    if (_history.length > 50) {
      _history = _history.take(50).toList();
    }
  }

  static void removeHistory(String id) {
    _history.removeWhere((item) => item.id == id);
  }

  static void clearHistory() {
    _history.clear();
  }
}

class SplashScreen extends StatefulWidget {
  @override
  _SplashScreenState createState() => _SplashScreenState();
}

class _SplashScreenState extends State<SplashScreen> with SingleTickerProviderStateMixin {
  late AnimationController _controller;
  late Animation<double> _fadeAnimation;
  late Animation<double> _scaleAnimation;

  @override
  void initState() {
    super.initState();
    _controller = AnimationController(
      duration: const Duration(seconds: 3),
      vsync: this,
    );
    
    _fadeAnimation = Tween<double>(begin: 0.0, end: 1.0).animate(
      CurvedAnimation(parent: _controller, curve: Curves.easeIn),
    );
    
    _scaleAnimation = Tween<double>(begin: 0.8, end: 1.0).animate(
      CurvedAnimation(parent: _controller, curve: Curves.elasticOut),
    );

    _controller.forward();
    
    Future.delayed(Duration(seconds: 4), () {
      Navigator.pushReplacement(
        context,
        MaterialPageRoute(builder: (context) => MissionInputPage()),
      );
    });
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.black,
      body: Center(
        child: AnimatedBuilder(
          animation: _controller,
          builder: (context, child) {
            return FadeTransition(
              opacity: _fadeAnimation,
              child: ScaleTransition(
                scale: _scaleAnimation,
                child: Column(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    Container(
                      padding: EdgeInsets.all(24),
                      decoration: BoxDecoration(
                        gradient: LinearGradient(
                          colors: [Colors.green.shade700, Colors.green.shade400],
                          begin: Alignment.topLeft,
                          end: Alignment.bottomRight,
                        ),
                        borderRadius: BorderRadius.circular(20),
                        boxShadow: [
                          BoxShadow(
                            color: Colors.green.withOpacity(0.3),
                            blurRadius: 20,
                            spreadRadius: 5,
                          ),
                        ],
                      ),
                      child: Text(
                        'uruti.rw',
                        style: GoogleFonts.interTight(
                          fontSize: 48,
                          fontWeight: FontWeight.bold,
                          color: Colors.white,
                          letterSpacing: -1.5,
                        ),
                      ),
                    ),
                    SizedBox(height: 30),
                    Text(
                      'Mission Classifier',
                      style: GoogleFonts.interTight(
                        fontSize: 24,
                        color: Colors.green.shade300,
                        letterSpacing: -0.8,
                      ),
                    ),
                    SizedBox(height: 50),
                    SpinKitThreeBounce(
                      color: Colors.green.shade400,
                      size: 30,
                    ),
                  ],
                ),
              ),
            );
          },
        ),
      ),
    );
  }
}

class MissionInputPage extends StatefulWidget {
  @override
  _MissionInputPageState createState() => _MissionInputPageState();
}

class _MissionInputPageState extends State<MissionInputPage> with SingleTickerProviderStateMixin {
  final FlutterSoundRecorder _recorder = FlutterSoundRecorder();
  final FlutterSoundPlayer _player = FlutterSoundPlayer();
  bool _isRecording = false;
  bool _isPlaying = false;
  String? _audioPath;
  late AnimationController _controller;
  late Animation<double> _animation;

  final TextEditingController _textController = TextEditingController();
  final String apiUrl = 'http://localhost:8080/predict';

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
    _initPlayer();
  }

  Future<void> _initRecorder() async {
    await Permission.microphone.request();
    await _recorder.openRecorder();
  }

  Future<void> _initPlayer() async {
    await _player.openPlayer();
  }

  Future<void> _startRecording() async {
    Directory tempDir = Directory.systemTemp;
    _audioPath = '${tempDir.path}/mission_input_${DateTime.now().millisecondsSinceEpoch}.wav';
    await _recorder.startRecorder(
      toFile: _audioPath,
      codec: Codec.pcm16WAV,
    );
    setState(() => _isRecording = true);
  }

  Future<void> _stopRecording() async {
    await _recorder.stopRecorder();
    setState(() => _isRecording = false);
    if (_audioPath != null) {
      _sendAudioToAPI(_audioPath!, 'audio');
    }
  }

  Future<void> _playRecording() async {
    if (_audioPath != null && File(_audioPath!).existsSync()) {
      if (_isPlaying) {
        await _player.stopPlayer();
        setState(() => _isPlaying = false);
      } else {
        await _player.startPlayer(
          fromURI: _audioPath,
          whenFinished: () {
            setState(() => _isPlaying = false);
          },
        );
        setState(() => _isPlaying = true);
      }
    }
  }

  Future<void> _pickAudioFile() async {
    FilePickerResult? result = await FilePicker.platform.pickFiles(
      type: FileType.audio,
      allowMultiple: false,
    );

    if (result != null) {
      String filePath = result.files.single.path!;
      String convertedPath = await _convertToWav(filePath);
      _sendAudioToAPI(convertedPath, 'upload');
    }
  }

  Future<String> _convertToWav(String inputPath) async {
    // For simplicity, we'll assume the conversion or just copy if it's already WAV
    // In a real app, you'd use FFmpeg or similar for actual conversion
    String fileName = path.basenameWithoutExtension(inputPath);
    String extension = path.extension(inputPath).toLowerCase();
    
    Directory tempDir = Directory.systemTemp;
    String outputPath = '${tempDir.path}/${fileName}_converted.wav';
    
    if (extension == '.wav') {
      // Already WAV, just copy
      await File(inputPath).copy(outputPath);
    } else {
      // For demo purposes, we'll just copy the file with .wav extension
      // In production, you'd use actual audio conversion
      await File(inputPath).copy(outputPath);
    }
    
    return outputPath;
  }

  Future<void> _sendTextToAPI() async {
    if (_textController.text.trim().isEmpty) {
      _showSnackBar('Please enter some text to analyze.');
      return;
    }

    Navigator.push(
      context,
      MaterialPageRoute(
        builder: (context) => ResultPage(
          future: _makeTextRequest(_textController.text.trim()),
          inputText: _textController.text.trim(),
          inputType: 'text',
        ),
      ),
    );
  }

  Future<Map<String, dynamic>> _makeTextRequest(String text) async {
    try {
      final response = await http.post(
        Uri.parse('$apiUrl?user_id=anonymous'),
        headers: {'Content-Type': 'application/json'},
        body: json.encode({'text': text}),
      );

      if (response.statusCode == 200) {
        return json.decode(response.body);
      } else {
        throw Exception('Failed to analyze text: ${response.statusCode}');
      }
    } catch (e) {
      throw Exception('Error connecting to API: $e');
    }
  }

  Future<void> _sendAudioToAPI(String audioPath, String inputType) async {
    Navigator.push(
      context,
      MaterialPageRoute(
        builder: (context) => ResultPage(
          future: _makeAudioRequest(audioPath),
          inputText: 'Audio input',
          audioPath: audioPath,
          inputType: inputType,
        ),
      ),
    );
  }

  Future<Map<String, dynamic>> _makeAudioRequest(String audioPath) async {
    try {
      var request = http.MultipartRequest('POST', Uri.parse('$apiUrl?user_id=anonymous'));
      
      request.files.add(await http.MultipartFile.fromPath(
        'file', 
        audioPath,
        contentType: MediaType('audio', 'wav'),
      ));
      
      var streamedResponse = await request.send();
      var response = await http.Response.fromStream(streamedResponse);

      if (response.statusCode == 200) {
        return json.decode(response.body);
      } else {
        throw Exception('Failed to analyze audio: ${response.statusCode}');
      }
    } catch (e) {
      throw Exception('Error connecting to API: $e');
    }
  }

  void _showSnackBar(String message) {
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: Text(message),
        backgroundColor: Colors.green.shade700,
        behavior: SnackBarBehavior.floating,
      ),
    );
  }

  void _navigateToHistory() {
    Navigator.push(
      context,
      MaterialPageRoute(builder: (context) => HistoryPage()),
    );
  }

  @override
  void dispose() {
    _recorder.closeRecorder();
    _player.closePlayer();
    _controller.dispose();
    _textController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text(
          'Mission Classifier',
          style: GoogleFonts.interTight(
            fontWeight: FontWeight.w600,
            letterSpacing: -0.8,
          ),
        ),
        backgroundColor: Colors.green.shade700,
        centerTitle: true,
        elevation: 0,
        actions: [
          IconButton(
            icon: Icon(Icons.history),
            onPressed: _navigateToHistory,
            tooltip: 'View History',
          ),
        ],
      ),
      body: SafeArea(
        child: SingleChildScrollView(
          padding: const EdgeInsets.all(20),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              // App Info Section
              Container(
                padding: EdgeInsets.all(20),
                decoration: BoxDecoration(
                  gradient: LinearGradient(
                    colors: [Colors.green.shade800, Colors.green.shade600],
                    begin: Alignment.topLeft,
                    end: Alignment.bottomRight,
                  ),
                  borderRadius: BorderRadius.circular(16),
                ),
                child: Column(
                  children: [
                    Icon(Icons.psychology, color: Colors.white, size: 48),
                    SizedBox(height: 12),
                    Text(
                      'AI-Powered Mission Analysis',
                      style: GoogleFonts.interTight(
                        fontSize: 20,
                        fontWeight: FontWeight.bold,
                        color: Colors.white,
                        letterSpacing: -0.8,
                      ),
                      textAlign: TextAlign.center,
                    ),
                    SizedBox(height: 8),
                    Text(
                      'Share your mission, vision, problems, or solutions through text, voice recording, or file upload for intelligent classification and insights.',
                      style: GoogleFonts.interTight(
                        fontSize: 14,
                        color: Colors.green.shade100,
                        letterSpacing: -0.4,
                      ),
                      textAlign: TextAlign.center,
                    ),
                  ],
                ),
              ),
              
              SizedBox(height: 24),
              
              // Text Input Section
              Text(
                'Text Input',
                style: GoogleFonts.interTight(
                  fontSize: 18,
                  fontWeight: FontWeight.w600,
                  color: Colors.green.shade300,
                  letterSpacing: -0.6,
                ),
              ),
              SizedBox(height: 12),
              TextField(
                controller: _textController,
                style: GoogleFonts.interTight(
                  letterSpacing: -0.6,
                  color: Colors.white,
                ),
                decoration: InputDecoration(
                  hintText: 'Describe your mission, vision, problem, or solution...',
                  hintStyle: TextStyle(color: Colors.grey.shade500),
                  filled: true,
                  fillColor: Colors.grey.shade900,
                  border: OutlineInputBorder(
                    borderRadius: BorderRadius.circular(12),
                    borderSide: BorderSide.none,
                  ),
                  focusedBorder: OutlineInputBorder(
                    borderRadius: BorderRadius.circular(12),
                    borderSide: BorderSide(color: Colors.green.shade400, width: 2),
                  ),
                ),
                maxLines: 5,
                minLines: 3,
              ),
              
              SizedBox(height: 16),
              
              ElevatedButton.icon(
                onPressed: _sendTextToAPI,
                icon: Icon(Icons.send, color: Colors.white),
                label: Text(
                  'Analyze Text',
                  style: GoogleFonts.interTight(
                    fontSize: 16,
                    fontWeight: FontWeight.w600,
                    letterSpacing: -0.6,
                    color: Colors.white,
                  ),
                ),
                style: ElevatedButton.styleFrom(
                  backgroundColor: Colors.green.shade700,
                  padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 16),
                  shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
                  elevation: 2,
                ),
              ),
              
              SizedBox(height: 32),
              
              // Divider
              Row(
                children: [
                  Expanded(child: Divider(color: Colors.grey.shade700)),
                  Padding(
                    padding: EdgeInsets.symmetric(horizontal: 16),
                    child: Text(
                      'OR',
                      style: GoogleFonts.interTight(
                        color: Colors.grey.shade500,
                        fontWeight: FontWeight.w500,
                      ),
                    ),
                  ),
                  Expanded(child: Divider(color: Colors.grey.shade700)),
                ],
              ),
              
              SizedBox(height: 32),
              
              // File Upload Section
              Text(
                'Upload Audio File',
                style: GoogleFonts.interTight(
                  fontSize: 18,
                  fontWeight: FontWeight.w600,
                  color: Colors.green.shade300,
                  letterSpacing: -0.6,
                ),
              ),
              SizedBox(height: 12),
              
              ElevatedButton.icon(
                onPressed: _pickAudioFile,
                icon: Icon(Icons.upload_file, color: Colors.white),
                label: Text(
                  'Upload Audio File',
                  style: GoogleFonts.interTight(
                    fontSize: 16,
                    fontWeight: FontWeight.w600,
                    letterSpacing: -0.6,
                    color: Colors.white,
                  ),
                ),
                style: ElevatedButton.styleFrom(
                  backgroundColor: Colors.blue.shade700,
                  padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 16),
                  shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
                  elevation: 2,
                ),
              ),
              
              SizedBox(height: 8),
              Text(
                'Supports: WAV, MP3, MP4, M4A, and other audio formats',
                style: GoogleFonts.interTight(
                  fontSize: 12,
                  color: Colors.grey.shade500,
                  letterSpacing: -0.4,
                ),
                textAlign: TextAlign.center,
              ),
              
              SizedBox(height: 32),
              
              // Divider
              Row(
                children: [
                  Expanded(child: Divider(color: Colors.grey.shade700)),
                  Padding(
                    padding: EdgeInsets.symmetric(horizontal: 16),
                    child: Text(
                      'OR',
                      style: GoogleFonts.interTight(
                        color: Colors.grey.shade500,
                        fontWeight: FontWeight.w500,
                      ),
                    ),
                  ),
                  Expanded(child: Divider(color: Colors.grey.shade700)),
                ],
              ),
              
              SizedBox(height: 32),
              
              // Voice Recording Section
              Text(
                'Voice Recording',
                style: GoogleFonts.interTight(
                  fontSize: 18,
                  fontWeight: FontWeight.w600,
                  color: Colors.green.shade300,
                  letterSpacing: -0.6,
                ),
              ),
              SizedBox(height: 20),
              
              Center(
                child: Column(
                  children: [
                    Container(
                      padding: EdgeInsets.all(20),
                      decoration: BoxDecoration(
                        shape: BoxShape.circle,
                        color: _isRecording ? Colors.green.shade700 : Colors.grey.shade900,
                        boxShadow: _isRecording ? [
                          BoxShadow(
                            color: Colors.green.withOpacity(0.4),
                            blurRadius: 20,
                            spreadRadius: 5,
                          ),
                        ] : null,
                      ),
                      child: _isRecording
                          ? ScaleTransition(
                              scale: _animation,
                              child: Icon(Icons.mic, color: Colors.white, size: 64),
                            )
                          : Icon(Icons.mic_none, color: Colors.white, size: 64),
                    ),
                    
                    SizedBox(height: 20),
                    
                    Row(
                      mainAxisAlignment: MainAxisAlignment.center,
                      children: [
                        ElevatedButton.icon(
                          onPressed: _isRecording ? _stopRecording : _startRecording,
                          icon: Icon(
                            _isRecording ? Icons.stop : Icons.mic,
                            color: Colors.white,
                          ),
                          label: Text(
                            _isRecording ? 'Stop & Analyze' : 'Start Recording',
                            style: GoogleFonts.interTight(
                              fontSize: 16,
                              fontWeight: FontWeight.w600,
                              letterSpacing: -0.6,
                              color: Colors.white,
                            ),
                          ),
                          style: ElevatedButton.styleFrom(
                            backgroundColor: _isRecording ? Colors.red.shade600 : Colors.green.shade700,
                            padding: const EdgeInsets.symmetric(horizontal: 30, vertical: 16),
                            shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
                            elevation: 3,
                          ),
                        ),
                        
                        if (_audioPath != null && File(_audioPath!).existsSync() && !_isRecording) ...[
                          SizedBox(width: 12),
                          ElevatedButton.icon(
                            onPressed: _playRecording,
                            icon: Icon(
                              _isPlaying ? Icons.stop : Icons.play_arrow,
                              color: Colors.white,
                            ),
                            label: Text(
                              _isPlaying ? 'Stop' : 'Play',
                              style: GoogleFonts.interTight(
                                fontSize: 14,
                                fontWeight: FontWeight.w600,
                                letterSpacing: -0.6,
                                color: Colors.white,
                              ),
                            ),
                            style: ElevatedButton.styleFrom(
                              backgroundColor: Colors.blue.shade600,
                              padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 16),
                              shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
                            ),
                          ),
                        ],
                      ],
                    ),
                    
                    if (_isRecording) ...[
                      SizedBox(height: 16),
                      Text(
                        'Recording in progress...',
                        style: GoogleFonts.interTight(
                          color: Colors.green.shade300,
                          fontSize: 14,
                          letterSpacing: -0.4,
                        ),
                      ),
                    ],
                  ],
                ),
              ),
              
              // Add extra padding at bottom for safe area
              SizedBox(height: 20),
            ],
          ),
        ),
      ),
    );
  }
}

class ResultPage extends StatelessWidget {
  final Future<Map<String, dynamic>> future;
  final String inputText;
  final String? audioPath;
  final String inputType;

  const ResultPage({
    super.key,
    required this.future,
    required this.inputText,
    this.audioPath,
    required this.inputType,
  });

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.black,
      appBar: AppBar(
        title: Text(
          'Analysis Results',
          style: GoogleFonts.interTight(
            fontWeight: FontWeight.w600,
            letterSpacing: -0.8,
          ),
        ),
        backgroundColor: Colors.green.shade700,
        elevation: 0,
      ),
      body: SafeArea(
        child: FutureBuilder<Map<String, dynamic>>(
          future: future,
          builder: (context, snapshot) {
            if (snapshot.connectionState == ConnectionState.waiting) {
              return Center(
                child: Padding(
                  padding: const EdgeInsets.all(24.0),
                  child: Column(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      SpinKitThreeBounce(color: Colors.green.shade400, size: 50),
                      SizedBox(height: 20),
                      Text(
                        'Analyzing your input...',
                        textAlign: TextAlign.center,
                        style: GoogleFonts.interTight(
                          fontSize: 20,
                          fontWeight: FontWeight.w500,
                          letterSpacing: -0.6,
                          color: Colors.white,
                        ),
                      ),
                      SizedBox(height: 8),
                      Text(
                        'This may take a few seconds.',
                        textAlign: TextAlign.center,
                        style: GoogleFonts.interTight(
                          fontSize: 14,
                          color: Colors.green.shade300,
                          letterSpacing: -0.4,
                        ),
                      ),
                    ],
                  ),
                ),
              );
            } else if (snapshot.hasError) {
              return Center(
                child: Padding(
                  padding: const EdgeInsets.all(24.0),
                  child: Column(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      Icon(Icons.error_outline, color: Colors.red, size: 64),
                      SizedBox(height: 20),
                      Text(
                        'Analysis Failed',
                        style: GoogleFonts.interTight(
                          fontSize: 24,
                          fontWeight: FontWeight.bold,
                          color: Colors.red,
                          letterSpacing: -0.8,
                        ),
                      ),
                      SizedBox(height: 12),
                      Text(
                        snapshot.error.toString(),
                        textAlign: TextAlign.center,
                        style: GoogleFonts.interTight(
                          fontSize: 14,
                          color: Colors.grey.shade400,
                          letterSpacing: -0.4,
                        ),
                      ),
                      SizedBox(height: 24),
                      ElevatedButton(
                        onPressed: () => Navigator.pop(context),
                        child: Text('Try Again'),
                        style: ElevatedButton.styleFrom(
                          backgroundColor: Colors.green.shade700,
                        ),
                      ),
                    ],
                  ),
                ),
              );
            } else {
              final result = snapshot.data!;
              
              // Save to history
              HistoryManager.addHistory(AnalysisHistory(
                id: DateTime.now().millisecondsSinceEpoch.toString(),
                inputText: inputText,
                audioPath: audioPath,
                result: result,
                timestamp: DateTime.now(),
                inputType: inputType,
              ));

              return SingleChildScrollView(
                padding: const EdgeInsets.all(20),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Container(
                      width: double.infinity,
                      padding: EdgeInsets.all(20),
                      decoration: BoxDecoration(
                        gradient: LinearGradient(
                          colors: [Colors.green.shade800, Colors.green.shade600],
                          begin: Alignment.topLeft,
                          end: Alignment.bottomRight,
                        ),
                        borderRadius: BorderRadius.circular(16),
                      ),
                      child: Column(
                        children: [
                          Icon(Icons.check_circle, color: Colors.white, size: 48),
                          SizedBox(height: 12),
                          Text(
                            'Analysis Complete',
                            style: GoogleFonts.interTight(
                              fontSize: 24,
                              fontWeight: FontWeight.bold,
                              color: Colors.white,
                              letterSpacing: -0.8,
                            ),
                          ),
                        ],
                      ),
                    ),
                    SizedBox(height: 24),
                    
                    // Display results
                    ...result.entries.map((entry) {
                      return Container(
                        margin: EdgeInsets.only(bottom: 16),
                        padding: EdgeInsets.all(16),
                        decoration: BoxDecoration(
                          color: Colors.grey.shade900,
                          borderRadius: BorderRadius.circular(12),
                        ),
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Text(
                              entry.key.toString().toUpperCase(),
                              style: GoogleFonts.interTight(
                                fontSize: 14,
                                fontWeight: FontWeight.w600,
                                color: Colors.green.shade300,
                                letterSpacing: -0.4,
                              ),
                            ),
                            SizedBox(height: 8),
                            Text(
                              entry.value.toString(),
                              style: GoogleFonts.interTight(
                                fontSize: 16,
                                color: Colors.white,
                                letterSpacing: -0.6,
                              ),
                            ),
                          ],
                        ),
                      );
                    }).toList(),
                    
                    SizedBox(height: 24),
                    
                    // Action buttons
                    Row(
                      children: [
                        Expanded(
                          child: ElevatedButton.icon(
                            onPressed: () {
                              Navigator.push(
                                context,
                                MaterialPageRoute(
                                  builder: (context) => RefineResultPage(
                                    originalResult: result,
                                    inputText: inputText,
                                  ),
                                ),
                              );
                            },
                            icon: Icon(Icons.auto_fix_high, color: Colors.white),
                            label: Text(
                              'Refine with AI',
                              style: GoogleFonts.interTight(
                                fontSize: 14,
                                fontWeight: FontWeight.w600,
                                letterSpacing: -0.6,
                                color: Colors.white,
                              ),
                            ),
                            style: ElevatedButton.styleFrom(
                              backgroundColor: Colors.purple.shade700,
                              padding: const EdgeInsets.symmetric(vertical: 12),
                              shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
                            ),
                          ),
                        ),
                        SizedBox(width: 12),
                        Expanded(
                          child: ElevatedButton.icon(
                            onPressed: () => Navigator.pop(context),
                            icon: Icon(Icons.refresh, color: Colors.white),
                            label: Text(
                              'Analyze Another',
                              style: GoogleFonts.interTight(
                                fontSize: 14,
                                fontWeight: FontWeight.w600,
                                letterSpacing: -0.6,
                                color: Colors.white,
                              ),
                            ),
                            style: ElevatedButton.styleFrom(
                              backgroundColor: Colors.green.shade700,
                              padding: const EdgeInsets.symmetric(vertical: 12),
                              shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
                            ),
                          ),
                        ),
                      ],
                    ),
                  ],
                ),
              );
            }
          },
        ),
      ),
    );
  }
}

class HistoryPage extends StatefulWidget {
  @override
  _HistoryPageState createState() => _HistoryPageState();
}

class _HistoryPageState extends State<HistoryPage> {
  final FlutterSoundPlayer _player = FlutterSoundPlayer();
  String? _currentlyPlayingId;

  @override
  void initState() {
    super.initState();
    _initPlayer();
  }

  Future<void> _initPlayer() async {
    await _player.openPlayer();
  }

  Future<void> _playAudio(String? audioPath, String id) async {
    if (audioPath == null || !File(audioPath).existsSync()) {
      _showSnackBar('Audio file not found');
      return;
    }

    if (_currentlyPlayingId == id) {
      await _player.stopPlayer();
      setState(() => _currentlyPlayingId = null);
    } else {
      await _player.stopPlayer();
      await _player.startPlayer(
        fromURI: audioPath,
        whenFinished: () {
          setState(() => _currentlyPlayingId = null);
        },
      );
      setState(() => _currentlyPlayingId = id);
    }
  }

  void _showSnackBar(String message) {
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: Text(message),
        backgroundColor: Colors.green.shade700,
        behavior: SnackBarBehavior.floating,
      ),
    );
  }

  void _deleteHistoryItem(String id) {
    setState(() {
      HistoryManager.removeHistory(id);
    });
    _showSnackBar('Item deleted from history');
  }

  void _clearAllHistory() {
    showDialog(
      context: context,
      builder: (BuildContext context) {
        return AlertDialog(
          backgroundColor: Colors.grey.shade900,
          title: Text(
            'Clear History',
            style: GoogleFonts.interTight(color: Colors.white),
          ),
          content: Text(
            'Are you sure you want to clear all history? This action cannot be undone.',
            style: GoogleFonts.interTight(color: Colors.white70),
          ),
          actions: [
            TextButton(
              onPressed: () => Navigator.pop(context),
              child: Text(
                'Cancel',
                style: GoogleFonts.interTight(color: Colors.green.shade300),
              ),
            ),
            TextButton(
              onPressed: () {
                HistoryManager.clearHistory();
                Navigator.pop(context);
                setState(() {});
                _showSnackBar('History cleared');
              },
              child: Text(
                'Clear',
                style: GoogleFonts.interTight(color: Colors.red.shade400),
              ),
            ),
          ],
        );
      },
    );
  }

  @override
  void dispose() {
    _player.closePlayer();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final history = HistoryManager.history;

    return Scaffold(
      backgroundColor: Colors.black,
      appBar: AppBar(
        title: Text(
          'Analysis History',
          style: GoogleFonts.interTight(
            fontWeight: FontWeight.w600,
            letterSpacing: -0.8,
          ),
        ),
        backgroundColor: Colors.green.shade700,
        elevation: 0,
        actions: [
          if (history.isNotEmpty)
            IconButton(
              icon: Icon(Icons.clear_all),
              onPressed: _clearAllHistory,
              tooltip: 'Clear All History',
            ),
        ],
      ),
      body: SafeArea(
        child: history.isEmpty
            ? Center(
                child: Column(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    Icon(
                      Icons.history,
                      size: 64,
                      color: Colors.grey.shade600,
                    ),
                    SizedBox(height: 16),
                    Text(
                      'No History Yet',
                      style: GoogleFonts.interTight(
                        fontSize: 20,
                        color: Colors.grey.shade400,
                        letterSpacing: -0.6,
                      ),
                    ),
                    SizedBox(height: 8),
                    Text(
                      'Your analysis results will appear here',
                      style: GoogleFonts.interTight(
                        fontSize: 14,
                        color: Colors.grey.shade600,
                        letterSpacing: -0.4,
                      ),
                    ),
                  ],
                ),
              )
            : ListView.builder(
                padding: EdgeInsets.all(16),
                itemCount: history.length,
                itemBuilder: (context, index) {
                  final item = history[index];
                  final isPlaying = _currentlyPlayingId == item.id;

                  return Card(
                    margin: EdgeInsets.only(bottom: 16),
                    color: Colors.grey.shade900,
                    child: Padding(
                      padding: EdgeInsets.all(16),
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          // Header with timestamp and type
                          Row(
                            mainAxisAlignment: MainAxisAlignment.spaceBetween,
                            children: [
                              Row(
                                children: [
                                  Icon(
                                    item.inputType == 'text'
                                        ? Icons.text_fields
                                        : item.inputType == 'audio'
                                            ? Icons.mic
                                            : Icons.upload_file,
                                    color: Colors.green.shade400,
                                    size: 20,
                                  ),
                                  SizedBox(width: 8),
                                  Text(
                                    item.inputType.toUpperCase(),
                                    style: GoogleFonts.interTight(
                                      fontSize: 12,
                                      fontWeight: FontWeight.w600,
                                      color: Colors.green.shade400,
                                      letterSpacing: -0.4,
                                    ),
                                  ),
                                ],
                              ),
                              Row(
                                children: [
                                  Text(
                                    '${item.timestamp.day}/${item.timestamp.month} ${item.timestamp.hour}:${item.timestamp.minute.toString().padLeft(2, '0')}',
                                    style: GoogleFonts.interTight(
                                      fontSize: 12,
                                      color: Colors.grey.shade500,
                                      letterSpacing: -0.4,
                                    ),
                                  ),
                                  SizedBox(width: 8),
                                  PopupMenuButton(
                                    color: Colors.grey.shade800,
                                    icon: Icon(Icons.more_vert, color: Colors.grey.shade400, size: 18),
                                    itemBuilder: (context) => [
                                      PopupMenuItem(
                                        value: 'delete',
                                        child: Row(
                                          children: [
                                            Icon(Icons.delete, color: Colors.red.shade400, size: 18),
                                            SizedBox(width: 8),
                                            Text(
                                              'Delete',
                                              style: GoogleFonts.interTight(color: Colors.white),
                                            ),
                                          ],
                                        ),
                                      ),
                                    ],
                                    onSelected: (value) {
                                      if (value == 'delete') {
                                        _deleteHistoryItem(item.id);
                                      }
                                    },
                                  ),
                                ],
                              ),
                            ],
                          ),
                          
                          SizedBox(height: 12),
                          
                          // Input preview
                          Container(
                            width: double.infinity,
                            padding: EdgeInsets.all(12),
                            decoration: BoxDecoration(
                              color: Colors.grey.shade800,
                              borderRadius: BorderRadius.circular(8),
                            ),
                            child: Column(
                              crossAxisAlignment: CrossAxisAlignment.start,
                              children: [
                                Text(
                                  'INPUT',
                                  style: GoogleFonts.interTight(
                                    fontSize: 10,
                                    fontWeight: FontWeight.w600,
                                    color: Colors.grey.shade400,
                                    letterSpacing: -0.2,
                                  ),
                                ),
                                SizedBox(height: 4),
                                if (item.inputType == 'text')
                                  Text(
                                    item.inputText.length > 100
                                        ? '${item.inputText.substring(0, 100)}...'
                                        : item.inputText,
                                    style: GoogleFonts.interTight(
                                      fontSize: 14,
                                      color: Colors.white,
                                      letterSpacing: -0.4,
                                    ),
                                  )
                                else
                                  Row(
                                    children: [
                                      Icon(Icons.audiotrack, color: Colors.green.shade400, size: 16),
                                      SizedBox(width: 8),
                                      Expanded(
                                        child: Text(
                                          'Audio recording',
                                          style: GoogleFonts.interTight(
                                            fontSize: 14,
                                            color: Colors.white,
                                            letterSpacing: -0.4,
                                          ),
                                        ),
                                      ),
                                      if (item.audioPath != null)
                                        IconButton(
                                          onPressed: () => _playAudio(item.audioPath, item.id),
                                          icon: Icon(
                                            isPlaying ? Icons.stop : Icons.play_arrow,
                                            color: Colors.green.shade400,
                                          ),
                                          iconSize: 20,
                                        ),
                                    ],
                                  ),
                              ],
                            ),
                          ),
                          
                          SizedBox(height: 12),
                          
                          // Results preview
                          Text(
                            'RESULTS',
                            style: GoogleFonts.interTight(
                              fontSize: 10,
                              fontWeight: FontWeight.w600,
                              color: Colors.grey.shade400,
                              letterSpacing: -0.2,
                            ),
                          ),
                          SizedBox(height: 8),
                          
                          ...item.result.entries.take(2).map((entry) {
                            return Padding(
                              padding: EdgeInsets.only(bottom: 8),
                              child: Row(
                                crossAxisAlignment: CrossAxisAlignment.start,
                                children: [
                                  Text(
                                    '${entry.key}: ',
                                    style: GoogleFonts.interTight(
                                      fontSize: 12,
                                      fontWeight: FontWeight.w600,
                                      color: Colors.green.shade400,
                                      letterSpacing: -0.4,
                                    ),
                                  ),
                                  Expanded(
                                    child: Text(
                                      entry.value.toString().length > 50
                                          ? '${entry.value.toString().substring(0, 50)}...'
                                          : entry.value.toString(),
                                      style: GoogleFonts.interTight(
                                        fontSize: 12,
                                        color: Colors.white,
                                        letterSpacing: -0.4,
                                      ),
                                    ),
                                  ),
                                ],
                              ),
                            );
                          }).toList(),
                          
                          if (item.result.length > 2)
                            Text(
                              '... and ${item.result.length - 2} more fields',
                              style: GoogleFonts.interTight(
                                fontSize: 11,
                                color: Colors.grey.shade500,
                                fontStyle: FontStyle.italic,
                                letterSpacing: -0.4,
                              ),
                            ),
                          
                          SizedBox(height: 12),
                          
                          // Action buttons
                          Row(
                            children: [
                              Expanded(
                                child: ElevatedButton.icon(
                                  onPressed: () {
                                    Navigator.push(
                                      context,
                                      MaterialPageRoute(
                                        builder: (context) => HistoryDetailPage(historyItem: item),
                                      ),
                                    );
                                  },
                                  icon: Icon(Icons.visibility, color: Colors.white, size: 16),
                                  label: Text(
                                    'View Details',
                                    style: GoogleFonts.interTight(
                                      fontSize: 12,
                                      fontWeight: FontWeight.w600,
                                      letterSpacing: -0.4,
                                      color: Colors.white,
                                    ),
                                  ),
                                  style: ElevatedButton.styleFrom(
                                    backgroundColor: Colors.blue.shade700,
                                    padding: const EdgeInsets.symmetric(vertical: 8),
                                    shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(8)),
                                  ),
                                ),
                              ),
                              SizedBox(width: 8),
                              Expanded(
                                child: ElevatedButton.icon(
                                  onPressed: () {
                                    Navigator.push(
                                      context,
                                      MaterialPageRoute(
                                        builder: (context) => RefineResultPage(
                                          originalResult: item.result,
                                          inputText: item.inputText,
                                        ),
                                      ),
                                    );
                                  },
                                  icon: Icon(Icons.auto_fix_high, color: Colors.white, size: 16),
                                  label: Text(
                                    'Refine',
                                    style: GoogleFonts.interTight(
                                      fontSize: 12,
                                      fontWeight: FontWeight.w600,
                                      letterSpacing: -0.4,
                                      color: Colors.white,
                                    ),
                                  ),
                                  style: ElevatedButton.styleFrom(
                                    backgroundColor: Colors.purple.shade700,
                                    padding: const EdgeInsets.symmetric(vertical: 8),
                                    shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(8)),
                                  ),
                                ),
                              ),
                            ],
                          ),
                        ],
                      ),
                    ),
                  );
                },
              ),
      ),
    );
  }
}

class HistoryDetailPage extends StatelessWidget {
  final AnalysisHistory historyItem;

  const HistoryDetailPage({super.key, required this.historyItem});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.black,
      appBar: AppBar(
        title: Text(
          'Analysis Details',
          style: GoogleFonts.interTight(
            fontWeight: FontWeight.w600,
            letterSpacing: -0.8,
          ),
        ),
        backgroundColor: Colors.green.shade700,
        elevation: 0,
      ),
      body: SafeArea(
        child: SingleChildScrollView(
          padding: const EdgeInsets.all(20),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              // Header
              Container(
                width: double.infinity,
                padding: EdgeInsets.all(20),
                decoration: BoxDecoration(
                  gradient: LinearGradient(
                    colors: [Colors.green.shade800, Colors.green.shade600],
                    begin: Alignment.topLeft,
                    end: Alignment.bottomRight,
                  ),
                  borderRadius: BorderRadius.circular(16),
                ),
                child: Column(
                  children: [
                    Icon(
                      historyItem.inputType == 'text'
                          ? Icons.text_fields
                          : historyItem.inputType == 'audio'
                              ? Icons.mic
                              : Icons.upload_file,
                      color: Colors.white,
                      size: 48,
                    ),
                    SizedBox(height: 12),
                    Text(
                      '${historyItem.inputType.toUpperCase()} Analysis',
                      style: GoogleFonts.interTight(
                        fontSize: 24,
                        fontWeight: FontWeight.bold,
                        color: Colors.white,
                        letterSpacing: -0.8,
                      ),
                    ),
                    SizedBox(height: 8),
                    Text(
                      '${historyItem.timestamp.day}/${historyItem.timestamp.month}/${historyItem.timestamp.year} at ${historyItem.timestamp.hour}:${historyItem.timestamp.minute.toString().padLeft(2, '0')}',
                      style: GoogleFonts.interTight(
                        fontSize: 14,
                        color: Colors.green.shade100,
                        letterSpacing: -0.4,
                      ),
                    ),
                  ],
                ),
              ),
              
              SizedBox(height: 24),
              
              // Input Section
              Text(
                'INPUT',
                style: GoogleFonts.interTight(
                  fontSize: 18,
                  fontWeight: FontWeight.w600,
                  color: Colors.green.shade300,
                  letterSpacing: -0.6,
                ),
              ),
              SizedBox(height: 12),
              Container(
                width: double.infinity,
                padding: EdgeInsets.all(16),
                decoration: BoxDecoration(
                  color: Colors.grey.shade900,
                  borderRadius: BorderRadius.circular(12),
                ),
                child: Text(
                  historyItem.inputText,
                  style: GoogleFonts.interTight(
                    fontSize: 16,
                    color: Colors.white,
                    letterSpacing: -0.6,
                  ),
                ),
              ),
              
              SizedBox(height: 24),
              
              // Results Section
              Text(
                'ANALYSIS RESULTS',
                style: GoogleFonts.interTight(
                  fontSize: 18,
                  fontWeight: FontWeight.w600,
                  color: Colors.green.shade300,
                  letterSpacing: -0.6,
                ),
              ),
              SizedBox(height: 12),
              
              ...historyItem.result.entries.map((entry) {
                return Container(
                  margin: EdgeInsets.only(bottom: 16),
                  padding: EdgeInsets.all(16),
                  decoration: BoxDecoration(
                    color: Colors.grey.shade900,
                    borderRadius: BorderRadius.circular(12),
                  ),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        entry.key.toString().toUpperCase(),
                        style: GoogleFonts.interTight(
                          fontSize: 14,
                          fontWeight: FontWeight.w600,
                          color: Colors.green.shade300,
                          letterSpacing: -0.4,
                        ),
                      ),
                      SizedBox(height: 8),
                      Text(
                        entry.value.toString(),
                        style: GoogleFonts.interTight(
                          fontSize: 16,
                          color: Colors.white,
                          letterSpacing: -0.6,
                        ),
                      ),
                    ],
                  ),
                );
              }).toList(),
              
              SizedBox(height: 24),
              
              // Action Button
              SizedBox(
                width: double.infinity,
                child: ElevatedButton.icon(
                  onPressed: () {
                    Navigator.push(
                      context,
                      MaterialPageRoute(
                        builder: (context) => RefineResultPage(
                          originalResult: historyItem.result,
                          inputText: historyItem.inputText,
                        ),
                      ),
                    );
                  },
                  icon: Icon(Icons.auto_fix_high, color: Colors.white),
                  label: Text(
                    'Refine with AI',
                    style: GoogleFonts.interTight(
                      fontSize: 16,
                      fontWeight: FontWeight.w600,
                      letterSpacing: -0.6,
                      color: Colors.white,
                    ),
                  ),
                  style: ElevatedButton.styleFrom(
                    backgroundColor: Colors.purple.shade700,
                    padding: const EdgeInsets.symmetric(vertical: 16),
                    shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
                  ),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

class RefineResultPage extends StatefulWidget {
  final Map<String, dynamic> originalResult;
  final String inputText;

  const RefineResultPage({
    super.key,
    required this.originalResult,
    required this.inputText,
  });

  @override
  _RefineResultPageState createState() => _RefineResultPageState();
}

class _RefineResultPageState extends State<RefineResultPage> {
  final TextEditingController _refineController = TextEditingController();
  final String apiUrl = 'http://localhost:8080/predict'; // Assuming same endpoint

  Future<Map<String, dynamic>> _refineWithAI(String refinementQuery) async {
    try {
      // Combine original input with refinement query
      String combinedInput = '''
Original analysis: ${widget.inputText}

Previous results: ${widget.originalResult.toString()}

Refinement request: $refinementQuery

Please provide a refined analysis based on this additional context.
''';

      final response = await http.post(
        Uri.parse('$apiUrl?user_id=anonymous'),
        headers: {'Content-Type': 'application/json'},
        body: json.encode({'text': combinedInput}),
      );

      if (response.statusCode == 200) {
        return json.decode(response.body);
      } else {
        throw Exception('Failed to refine analysis: ${response.statusCode}');
      }
    } catch (e) {
      throw Exception('Error connecting to API: $e');
    }
  }

  void _submitRefinement() {
    if (_refineController.text.trim().isEmpty) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text('Please enter your refinement request'),
          backgroundColor: Colors.red.shade700,
          behavior: SnackBarBehavior.floating,
        ),
      );
      return;
    }

    Navigator.push(
      context,
      MaterialPageRoute(
        builder: (context) => ResultPage(
          future: _refineWithAI(_refineController.text.trim()),
          inputText: 'Refined: ${_refineController.text.trim()}',
          inputType: 'refined',
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.black,
      appBar: AppBar(
        title: Text(
          'Refine with AI',
          style: GoogleFonts.interTight(
            fontWeight: FontWeight.w600,
            letterSpacing: -0.8,
          ),
        ),
        backgroundColor: Colors.purple.shade700,
        elevation: 0,
      ),
      body: SafeArea(
        child: SingleChildScrollView(
          padding: const EdgeInsets.all(20),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              // Header
              Container(
                width: double.infinity,
                padding: EdgeInsets.all(20),
                decoration: BoxDecoration(
                  gradient: LinearGradient(
                    colors: [Colors.purple.shade800, Colors.purple.shade600],
                    begin: Alignment.topLeft,
                    end: Alignment.bottomRight,
                  ),
                  borderRadius: BorderRadius.circular(16),
                ),
                child: Column(
                  children: [
                    Icon(Icons.auto_fix_high, color: Colors.white, size: 48),
                    SizedBox(height: 12),
                    Text(
                      'AI-Powered Refinement',
                      style: GoogleFonts.interTight(
                        fontSize: 24,
                        fontWeight: FontWeight.bold,
                        color: Colors.white,
                        letterSpacing: -0.8,
                      ),
                      textAlign: TextAlign.center,
                    ),
                    SizedBox(height: 8),
                    Text(
                      'Ask specific questions about your analysis or request improvements to get more detailed insights.',
                      style: GoogleFonts.interTight(
                        fontSize: 14,
                        color: Colors.purple.shade100,
                        letterSpacing: -0.4,
                      ),
                      textAlign: TextAlign.center,
                    ),
                  ],
                ),
              ),
              
              SizedBox(height: 24),
              
              // Original Results Summary
              Text(
                'ORIGINAL ANALYSIS',
                style: GoogleFonts.interTight(
                  fontSize: 16,
                  fontWeight: FontWeight.w600,
                  color: Colors.green.shade300,
                  letterSpacing: -0.6,
                ),
              ),
              SizedBox(height: 12),
              Container(
                width: double.infinity,
                padding: EdgeInsets.all(16),
                decoration: BoxDecoration(
                  color: Colors.grey.shade900,
                  borderRadius: BorderRadius.circular(12),
                  border: Border.all(color: Colors.green.shade700, width: 1),
                ),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      'Input: ${widget.inputText}',
                      style: GoogleFonts.interTight(
                        fontSize: 14,
                        color: Colors.white70,
                        letterSpacing: -0.4,
                      ),
                    ),
                    SizedBox(height: 12),
                    ...widget.originalResult.entries.take(3).map((entry) {
                      return Padding(
                        padding: EdgeInsets.only(bottom: 8),
                        child: RichText(
                          text: TextSpan(
                            children: [
                              TextSpan(
                                text: '${entry.key}: ',
                                style: GoogleFonts.interTight(
                                  fontSize: 14,
                                  fontWeight: FontWeight.w600,
                                  color: Colors.green.shade400,
                                  letterSpacing: -0.4,
                                ),
                              ),
                              TextSpan(
                                text: entry.value.toString().length > 80
                                    ? '${entry.value.toString().substring(0, 80)}...'
                                    : entry.value.toString(),
                                style: GoogleFonts.interTight(
                                  fontSize: 14,
                                  color: Colors.white,
                                  letterSpacing: -0.4,
                                ),
                              ),
                            ],
                          ),
                        ),
                      );
                    }).toList(),
                    if (widget.originalResult.length > 3)
                      Text(
                        '... and ${widget.originalResult.length - 3} more fields',
                        style: GoogleFonts.interTight(
                          fontSize: 12,
                          color: Colors.grey.shade500,
                          fontStyle: FontStyle.italic,
                          letterSpacing: -0.4,
                        ),
                      ),
                  ],
                ),
              ),
              
              SizedBox(height: 24),
              
              // Refinement Input
              Text(
                'REFINEMENT REQUEST',
                style: GoogleFonts.interTight(
                  fontSize: 16,
                  fontWeight: FontWeight.w600,
                  color: Colors.purple.shade300,
                  letterSpacing: -0.6,
                ),
              ),
              SizedBox(height: 12),
              TextField(
                controller: _refineController,
                style: GoogleFonts.interTight(
                  letterSpacing: -0.6,
                  color: Colors.white,
                ),
                decoration: InputDecoration(
                  hintText: 'Ask for more details, clarification, or specific aspects you want to explore...\n\nExamples:\n• Can you provide more specific recommendations?\n• What are the potential risks involved?\n• How can this be implemented practically?',
                  hintStyle: TextStyle(color: Colors.grey.shade500),
                  filled: true,
                  fillColor: Colors.grey.shade900,
                  border: OutlineInputBorder(
                    borderRadius: BorderRadius.circular(12),
                    borderSide: BorderSide.none,
                  ),
                  focusedBorder: OutlineInputBorder(
                    borderRadius: BorderRadius.circular(12),
                    borderSide: BorderSide(color: Colors.purple.shade400, width: 2),
                  ),
                ),
                maxLines: 6,
                minLines: 4,
              ),
              
              SizedBox(height: 24),
              
              // Submit Button
              SizedBox(
                width: double.infinity,
                child: ElevatedButton.icon(
                  onPressed: _submitRefinement,
                  icon: Icon(Icons.psychology, color: Colors.white),
                  label: Text(
                    'Refine Analysis',
                    style: GoogleFonts.interTight(
                      fontSize: 16,
                      fontWeight: FontWeight.w600,
                      letterSpacing: -0.6,
                      color: Colors.white,
                    ),
                  ),
                  style: ElevatedButton.styleFrom(
                    backgroundColor: Colors.purple.shade400,
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(12),
                    ),
                  ),
                ),
              ),
            ],
          ),
        ),
      ),
    );  }
}

