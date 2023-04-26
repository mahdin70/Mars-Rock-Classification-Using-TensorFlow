import * as tf from '@tensorflow/tfjs';
import * as tmImage from '@teachablemachine/image';

const URL = 'https://teachablemachine.withgoogle.com/models/your-model-url/';
const modelURL = URL + 'model.json';
const metadataURL = URL + 'metadata.json';

const modelPromise = tmImage.load(modelURL, metadataURL);

// Get the video element from the HTML document
const video = document.getElementById('rover-camera');

// Request access to the camera and stream video
navigator.mediaDevices.getUserMedia({ video: true })
  .then(function(stream) {
    video.srcObject = stream;
    video.play();
  })
  .catch(function(error) {
    console.log(error);
  });

// Make a prediction every time a new frame is available
video.addEventListener('loadeddata', async function() {
  const model = await modelPromise;
  const tensor = tmImage.capture(video).expandDims().div(255);
  const prediction = await model.predict(tensor).data();

  // Get the class with the highest probability
  const classIndex = prediction.indexOf(Math.max(...prediction));

  // Get the label for the predicted class
  const metadata = await model.getMetadata();
  const label = metadata.labels[classIndex];

  console.log(label);
});
