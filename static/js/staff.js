async function updateStatus(complaintId) {
  const newStatus = prompt(
    "Enter new status (submitted/in_progress/resolved):"
  );
  if (!newStatus) return;

  try {
    const response = await fetch(`/update_status/${complaintId}`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ status: newStatus }),
    });

    if (response.ok) {
      location.reload();
    } else {
      alert("Error updating status. Please try again.");
    }
  } catch (error) {
    console.error("Error:", error);
    alert("Error updating status. Please try again.");
  }
}

function viewDetails(complaintId) {
  window.location.href = `/complaint_details/${complaintId}`;
}
