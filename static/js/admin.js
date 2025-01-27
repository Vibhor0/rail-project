function showAddStaffForm() {
  // Implementation for showing add staff form modal
  console.log("Show add staff form");
}

async function editStaff(staffId) {
  // Implementation for editing staff
  console.log("Edit staff:", staffId);
}

async function removeStaff(staffId) {
  if (!confirm("Are you sure you want to remove this staff member?")) return;

  try {
    const response = await fetch(`/remove_staff/${staffId}`, {
      method: "POST",
    });

    if (response.ok) {
      location.reload();
    } else {
      alert("Error removing staff member. Please try again.");
    }
  } catch (error) {
    console.error("Error:", error);
    alert("Error removing staff member. Please try again.");
  }
}
