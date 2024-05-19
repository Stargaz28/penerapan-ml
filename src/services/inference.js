const tf = require('@tensorflow/tfjs-node');
const InputError = require('../exceptions/InputError');

async function predictClassificationModel(model, image){
  try {
    const tensor = tf.node 
      .decodeImage(image)
      .resizeNearestNeighbor([224,224])
      .expandDims()
      .toFloat()

      const prediction = model.predict(tensor);
      const score = await prediction.data();
      const confidenceScore = score[0] * 100; 

    const label = confidenceScore > 50 ? 'Cancer' : 'Non-cancer';

    let suggestion;
    if (label === 'Cancer') {
      suggestion = "Segera periksa ke dokter!";
    } else {
      suggestion = "Anda sehat!";
    }

    return { label, confidenceScore, suggestion };
  } catch (error) {
    throw new InputError('Terjadi kesalahan dalam melakukan prediksi');
  }
}

module.exports = predictClassificationModel;