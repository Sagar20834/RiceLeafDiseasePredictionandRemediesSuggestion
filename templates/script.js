// script.js

// Add a click event listener to both buttons
document.getElementById("ok-button").addEventListener("click", refreshPage);
document.getElementById("cancel-button").addEventListener("click", refreshPage);

function refreshPage() {
    // Refresh the page when either button is clicked
    window.location.reload();
}
