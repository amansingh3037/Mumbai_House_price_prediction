function getData() {
    fetch("/get-data")
    .then(response => response.json())
    .then(data => {

        document.getElementById("longitude").value = data.longitude;
        document.getElementById("latitude").value = data.latitude;
        document.getElementById("housing_median_age").value = data.housing_median_age;
        document.getElementById("total_rooms").value = data.total_rooms;
        document.getElementById("total_bedrooms").value = data.total_bedrooms;
        document.getElementById("population").value = data.population;
        document.getElementById("households").value = data.households;
        document.getElementById("median_income").value = data.median_income;
        document.getElementById("ocean_proximity").value = data.ocean_proximity;

        document.getElementById("expected_prediction").value = data.median_house_value;
    });
}

function predict() {

    const data = {
        longitude: Number(document.getElementById("longitude").value),
        latitude: Number(document.getElementById("latitude").value),
        housing_median_age: Number(document.getElementById("housing_median_age").value),
        total_rooms: Number(document.getElementById("total_rooms").value),
        total_bedrooms: Number(document.getElementById("total_bedrooms").value),
        population: Number(document.getElementById("population").value),
        households: Number(document.getElementById("households").value),
        median_income: Number(document.getElementById("median_income").value),
        ocean_proximity: document.getElementById("ocean_proximity").value
    };

    fetch("/predict", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify(data)
    })
    .then(response => response.json())
    .then(result => {
        document.getElementById("actual_prediction").value = result.prediction;
    });
}