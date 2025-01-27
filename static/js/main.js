document
  .getElementById("complaintForm")
  .addEventListener("submit", async (e) => {
    e.preventDefault();

    const formData = new FormData(e.target);

    try {
      const response = await fetch("/submit_complaint", {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (response.ok) {
        alert(
          `Complaint submitted successfully!\nComplaint ID: ${data.complaint_id}\nCategory: ${data.category}`
        );
        e.target.reset();
      } else {
        alert("Error submitting complaint. Please try again.");
      }
    } catch (error) {
      console.error("Error:", error);
      alert("Error submitting complaint. Please try again.");
    }
  });

document
  .getElementById("trackingForm")
  .addEventListener("submit", async (e) => {
    e.preventDefault();

    const complaintId = document.getElementById("complaintId").value;
    window.location.href = `/track_complaint/${complaintId}`;
  });
