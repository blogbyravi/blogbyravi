const contactForm = document.querySelector('#contact-form');
const formStatus = document.querySelector('#form-status');
const submitButton = contactForm.querySelector('button[type="submit"]');
const mainNav = document.querySelector('.main-nav');
const menuToggle = document.querySelector('.menu-toggle');

contactForm.addEventListener('submit', async (event) => {
  event.preventDefault();
  submitButton.disabled = true;
  submitButton.innerHTML = 'Sending…';
  formStatus.textContent = '';

  try {
    const response = await fetch(contactForm.action, {
      method: 'POST',
      body: new FormData(contactForm),
      headers: { Accept: 'application/json' }
    });
    const result = await response.json();
    if (!response.ok || String(result.success).toLowerCase() !== 'true') {
      throw new Error(result.message || 'Message could not be sent.');
    }

    contactForm.reset();
    formStatus.textContent = /activat/i.test(result.message || '')
      ? 'Please confirm the activation email sent to the inbox before messages can be delivered.'
      : 'Thanks, your message has been sent.';
  } catch (error) {
    formStatus.textContent = 'We could not send your message just now. Please email ravikirankc@zohomail.in directly.';
  } finally {
    submitButton.disabled = false;
    submitButton.innerHTML = 'Send message <span aria-hidden="true">↗</span>';
  }
});

menuToggle.addEventListener('click', () => {
  const open = menuToggle.getAttribute('aria-expanded') !== 'true';
  menuToggle.setAttribute('aria-expanded', String(open));
  mainNav.classList.toggle('is-open', open);
});

mainNav.querySelectorAll('a').forEach((link) => {
  link.addEventListener('click', () => {
    mainNav.classList.remove('is-open');
    menuToggle.setAttribute('aria-expanded', 'false');
  });
});

document.querySelector('#year').textContent = new Date().getFullYear();
