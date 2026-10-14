document.getElementById("predictionForm")
.addEventListener("submit", async function(e){

    e.preventDefault();

    const age = document.getElementById("age").value;
    const income = document.getElementById("income").value;

    if(age <= 0 || income <= 0){
        alert("Please enter valid values");
        return;
    }

    const formData = new FormData();
    formData.append("age", age);
    formData.append("income", income);

    const response = await fetch("/predict", {
        method: "POST",
        body: formData
    });

    const data = await response.json();

    document.getElementById("result").innerHTML =
        data.prediction;
});