document.querySelector("form").addEventListener("submit", async (event) => {
  event.preventDefault();

  const question = document.querySelector(
    'input[name="question"]'
  ).value;
  const loadingDiv = document.querySelector(".dot-pulse");
  const resultDiv = document.querySelector(".result");

  loadingDiv.style.display = "block";
  resultDiv.style.display = "none";

  try {
    const response = await fetch("http://127.0.0.1:5000/gpt", {
      method: "POST",
      body: JSON.stringify({ question: question }),
      headers: {
        "Content-Type": "application/json", // Set the content type to JSON
      },
    });

    if (!response.ok) {
      throw new Error("Request failed");
    }

    const data = await response.json();
    const answer = data.answer;

    resultDiv.textContent = answer;
  } catch (error) {
    resultDiv.textContent = error.message;
  } finally {
    loadingDiv.style.display = "none";
    resultDiv.style.display = "block";
  }
});
