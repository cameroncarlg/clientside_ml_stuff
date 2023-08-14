console.log('Hello TensorFlow!')

/**
 * Get the car data reduced to just the variables we are interested in
 * and cleaned of missing data.
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
}

document.addEventListener('DOMContentLoaded', run);