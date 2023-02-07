import 'dart:async';
import 'dart:convert';
import 'dart:typed_data';

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:geolocator/geolocator.dart';
import 'package:http/http.dart' as http;

Future<void> main() async {
  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: Home(),
    );
  }
}

class Home extends StatefulWidget {
  @override
  _HomeState createState() => _HomeState();
}

class _HomeState extends State<Home> {
  List<CameraDescription>? cameras; //list out the camera available
  CameraController? controller; //controller for camera
  XFile? image; //for caputred image
  Position? _position;
  Timer? _timer;

  @override
  void initState() {
    loadCamera();
    super.initState();
  }

  loadCamera() async {
    cameras = await availableCameras();
    if (cameras != null) {
      controller = CameraController(cameras![0], ResolutionPreset.max);
      controller!.initialize().then((_) {
        if (!mounted) {
          return;
        }
        setState(() {});
      });
    } else {
      print("NO any camera found");
    }
  }

  void _getCurrentLocation() async {
    Position position = await _determinePosition();
    setState(() {
      _position = position;
    });
  }

  Future<Position> _determinePosition() async {
    LocationPermission permission;

    permission = await Geolocator.checkPermission();

    if (permission == LocationPermission.denied) {
      permission = await Geolocator.requestPermission();
      if (permission == LocationPermission.denied) {
        return Future.error('Location Permissions are denied');
      }
    }

    return await Geolocator.getCurrentPosition();
  }

  void myImageCapture() async {
    try {
      if (controller != null) {
        //check if contrller is not null
        if (controller!.value.isInitialized) {
          //check if controller is initialized
          image = await controller!.takePicture(); //capture image
          setState(() {});
        }
      }
    } catch (e) {
      print(e); //show error
    }
  }

  Future<http.Response> postRequest() async {
    var url = Uri.parse('http://118.67.129.236:30011/OD');

    Uint8List fileBytes = await image!.readAsBytes();
    String fileBase64 = base64Encode(fileBytes);

    Map data = {
      'files': fileBase64,
      'lat': _position!.latitude,
      'lon': _position!.longitude
    };
    var body = json.encode(data);

    var response = await http.post(url,
        headers: {'Content-Type': 'application/json'}, body: body);

    return response;
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
        appBar: AppBar(
          title: Text("Live Camera Frame"),
          backgroundColor: Color.fromARGB(255, 82, 82, 255),
        ),
        body: Container(
            child: Column(children: [
          Container(
              height: 480,
              width: 270,
              child: controller == null
                  ? Center(child: Text("Loading Camera..."))
                  : !controller!.value.isInitialized
                      ? Center(
                          child: CircularProgressIndicator(),
                        )
                      : CameraPreview(controller!)),
          Container(
              height: 60,
              padding: EdgeInsets.only(top: 20, bottom: 10),
              child: Text("${_position?.latitude}, ${_position?.longitude}")),
          Container(
              child:
                  Row(mainAxisAlignment: MainAxisAlignment.center, children: [
            Container(
                padding: EdgeInsets.only(right: 30),
                child: OutlinedButton(
                    child: Text("Capture Start"),
                    onPressed: () async {
                      _timer =
                          Timer.periodic(Duration(seconds: 3), (timer) async {
                        _getCurrentLocation();
                        myImageCapture();
                        postRequest();
                      });
                    })),
            Container(
                padding: EdgeInsets.only(left: 30),
                child: OutlinedButton(
                    child: Text("Capture End"),
                    onPressed: () {
                      _timer!.cancel();
                    }))
          ]))
        ])));
  }
}
