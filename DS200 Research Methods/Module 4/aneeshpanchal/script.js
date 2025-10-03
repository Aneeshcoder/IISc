// Enhanced JS for Smooth Section Highlights and Scroll Behavior

document.addEventListener('DOMContentLoaded', () => {
  const sections = document.querySelectorAll('main section');
  const navLinks = document.querySelectorAll('nav a');

  // Smooth scroll on nav click
  navLinks.forEach(link => {
    if (link.getAttribute('href').startsWith('#')) {
      link.addEventListener('click', e => {
        e.preventDefault();
        const targetId = link.getAttribute('href').substring(1);
        const targetSection = document.getElementById(targetId);
        if (targetSection) {
          window.scrollTo({
            top: targetSection.offsetTop - 90,
            behavior: 'smooth'
          });
        }
      });
    }
  });

  // Highlight nav item on scroll
  window.addEventListener('scroll', () => {
    let currentSectionId = '';
    sections.forEach(section => {
      if (window.pageYOffset >= section.offsetTop - 120) {
        currentSectionId = section.id;
      }
    });
    navLinks.forEach(link => {
      link.classList.remove('active');
      if (link.getAttribute('href') === '#' + currentSectionId) {
        link.classList.add('active');
      }
    });
  });
});
