const togglePassword = document.querySelector('#togglepassword');
const confirm_togglepassword = document.querySelector('#confirm-togglepassword');
const password = document.querySelector('#password');
const confirm_password = document.querySelector('#confirm-password');

togglePassword.addEventListener('click', function (e) {
    const type = password.getAttribute('type') === 'password' ? 'text' : 'password';
    password.setAttribute('type', type);

    // Toggle the eye icon
    this.classList.toggle('fa-eye');
    this.classList.toggle('fa-eye-slash');
});

confirm_togglepassword.addEventListener('click', function (e) {
    const type = confirm_password.getAttribute('type') === 'password' ? 'text' : 'password';
    confirm_password.setAttribute('type', type);

    // Toggle the eye icon
    this.classList.toggle('fa-eye');
    this.classList.toggle('fa-eye-slash');
});
