const basicContainer = document.querySelector('.basic-container');
const proContainer = document.querySelector('.pro-container');
const switchBtn = document.querySelector('#switch-btn');
const predictBtn = document.querySelector('.basic-container .btn-primary');

let isBasicModeUsed = false;

switchBtn.addEventListener('click', () => {
  basicContainer.classList.toggle('d-none');
  proContainer.classList.toggle('d-none');

  if (basicContainer.classList.contains('d-none')) {
    switchBtn.textContent = 'Switch to Basic';
    proContainer.classList.remove('d-none');
  } else {
    switchBtn.textContent = 'Switch to Pro';
  }
});

predictBtn.addEventListener('click', () => {
  if (basicContainer.classList.contains('d-none')) {
    if (!isBasicModeUsed) {

      // Set the flag to indicate that basic mode has been used
      isBasicModeUsed = true;

      // Disable the predict button
      predictBtn.disabled = true;

      // Notify the user about the cooldown period
      alert('Thank you for trying our Basic mode feature. There is a cooldown period of 24 hours before you can use it again.');

      // Set a timeout to enable the button after 24 hours
      setTimeout(() => {
        predictBtn.disabled = false;
        isBasicModeUsed = false;
      }, 24 * 60 * 60 * 1000); // 24 hours in milliseconds
    } else {
      alert('You have already used the basic mode. Please wait for the cooldown period to end.');
    }
  }
});
