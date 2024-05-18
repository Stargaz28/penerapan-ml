const tf = require('@tensorflow/tfjs-node');

async function predictClassificationModel(model, image){
    const tensor = tf.node 
    .decodeImage(image)
    .resizeNearestNeighbor([224,224])
    .expandDims()
    .toFloat()

    const prediction = model.predict(tensor);
    const score = await prediction.data();
    const confidenceScore = score[0] * 100; // Hanya satu nilai karena binary classification

  const label = confidenceScore > 50 ? 'Cancer' : 'Non-cancer';

  let suggestion;
  if (label === 'Cancer') {
    suggestion = "Segera periksa ke dokter!";
  } else {
    suggestion = "Anda sehat!";
  }

  return { label, confidenceScore, suggestion };
}

module.exports = predictClassificationModel;