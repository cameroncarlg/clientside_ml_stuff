/**
 * Get the car data reduced to just the variables we are interested in
 * and clean the missing data.
 */

const getData = async () => {
    // fetch/json appropriate car data
    const carsDataResponse = await fetch('https://storage.googleapis.com/tfjs-tutorials/carsData.json')
    const carsData = await carsDataResponse.json();

    console.log('yayy', carsData)

    // clean fetched data, map over the json and filter out entries missing mpg and horsepower
    const cleaned = carsData.map(car => ({
        mpg: car.Miles_per_Gallon,
        horsepower: car.Horsepower,
    }))
    .filter(car => (car.mpg != null && car.horsepower != null));

    return cleaned
}

const createModel = () => {
    // create a sequential model
    const model = tf.sequential();

    // add a single input layer
    model.add(tf.layers.dense({inputShape: [1], units: 1, useBias: true}));

    // add an output layer
    model.add(tf.layers.dense({units: 1, useBias: true}))

    return model;
}

/**
 * convert the input data to tensors that we can use for machine learning.
 * we will also do the important best practices of _shuffling_ the data
 * and _normalizing_ the data
 * MPG on the y-axis
 */

const convertToTensor = data => tf.tidy(() => {
    // shuffle the data
    tf.util.shuffle(data)

    // convert data into tensors
    const inputs = data.map(d => d.horsepower)
    const labels = data.map(d => d.mpg)
    const inputTensor = tf.tensor2d(inputs, [inputs.length, 1])
    const labelTensor = tf.tensor2d(labels, [labels.length, 1])

    // normalize the data using min-max scaling
    const inputMax = inputTensor.max();
    const inputMin = inputTensor.min();
    const labelMax = inputTensor.max();
    const labelMin = inputTensor.min();

    const normalizedInputs = inputTensor.sub(inputMin).div(inputMax.sub(inputMin));
    const normalizedLabels = labelTensor.sub(labelMin).div(labelMax.sub(labelMin));

    return {
        inputs: normalizedInputs,
        labels: normalizedLabels,
        inputMax,
        inputMin,
        labelMax,
        labelMin,
      };
});

const trainModel = async (model, inputs, labels) => {
    model.compile({
        optimizer: tf.train.adam(),
        loss: tf.losses.meanSquaredError,
        metrics: ['mse'],
    })

    const batchSize = 32;
    const epochs = 50;

    return await model.fit(inputs, labels, {
        batchSize,
        epochs,
        shuffle: true,
        callbacks: tfvis.show.fitCallbacks(
            { name: 'Training Performance' },
            ['loss', 'mse'],
            { height: 300, callbacks: ['onEpochEnd'] }
        )
    })
}

const testModel = (model, inputData, normalizationData) => {
    const { inputMax, inputMin, labelMin, labelMax } = normalizationData;

    const [xs, preds] = tf.tidy(() => {
        // create some predictions
        const xsNorm = tf.linspace(0, 1, 100);
        const predictions = model.predict(xsNorm.reshape([100, 1]));

        // un-normalize the data
        const unNormXs = xsNorm.mul(inputMax.sub(inputMin)).add(inputMin);
        const unNormPreds = predictions.mul(labelMax.sub(labelMin)).add(labelMin);

        return [unNormXs.dataSync(), unNormPreds.dataSync()];
    })

    const predictedPoints = Array.from(xs).map((val, i) => ({
        x: val,
        y: preds[i]
    }));

    const originalPoints = inputData.map(d => ({
        x: d.horsepower,
        y: d.mpg,
    }));

    tfvis.render.scatterplot(
        { name: 'Model Predictions vs Original Data'},
        { values: [originalPoints, predictedPoints], series: ['original', 'predicted'] },
        {
            xLabel: 'Horsepower',
            yLabel: 'MPG',
            height: 300
        }
    )
}

const run = async () => {
    // Load and plot the original data we are going to train on
    const data = await getData();
    const values = data.map(d => ({
        x: d.horsepower,
        y: d.mpg
    }));

    tfvis.render.scatterplot(
        {name: 'Horsepower vs MPG'},
        {values},
        {
            xLabel: 'Horsepower',
            yLabel: 'MPG',
            height: 300
        }
    );

    // More code will be added here below
    // Create the model
    const model = createModel();
    tfvis.show.modelSummary({name: 'Model Summary'}, model);

    // Convert the data to a form we can use for training
    const tensorData = convertToTensor(data);
    const {inputs, labels} = tensorData;

    // Train
    await trainModel(model, inputs, labels);
    console.log('Done Training');

    // Make some predictions using the model and compare them to the
    // original data
    testModel(model, data, tensorData);
}



document.addEventListener('DOMContentLoaded', run);