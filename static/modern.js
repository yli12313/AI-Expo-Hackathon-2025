// Tab switching functionality
document.addEventListener('DOMContentLoaded', function() {
    const tabBtns = document.querySelectorAll('.tab-btn');
    const tabPanels = document.querySelectorAll('.tab-panel');

    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const targetTab = btn.getAttribute('data-tab');
            
            // Remove active class from all tabs and panels
            tabBtns.forEach(b => b.classList.remove('active'));
            tabPanels.forEach(p => p.classList.remove('active'));
            
            // Add active class to clicked tab and corresponding panel
            btn.classList.add('active');
            document.getElementById(targetTab + '-panel').classList.add('active');
        });
    });

    // Floating particles animation
    function createParticle() {
        const particlesContainer = document.querySelector('.floating-particles');
        if (!particlesContainer) return;

        const particle = document.createElement('div');
        particle.className = 'particle';
        particle.style.left = Math.random() * 100 + '%';
        particle.style.animationDuration = (Math.random() * 3 + 2) + 's';
        particle.style.opacity = Math.random() * 0.5 + 0.2;
        particlesContainer.appendChild(particle);

        setTimeout(() => {
            if (particle.parentNode) {
                particle.remove();
            }
        }, 5000);
    }

    // Create particles periodically
    setInterval(createParticle, 300);

    // Add some initial particles
    for (let i = 0; i < 5; i++) {
        setTimeout(createParticle, i * 100);
    }

    // Add smooth scroll behavior for better UX
    document.documentElement.style.scrollBehavior = 'smooth';

    // Add keyboard navigation for tabs
    document.addEventListener('keydown', function(e) {
        if (e.key === 'ArrowRight' || e.key === 'ArrowLeft') {
            const activeTab = document.querySelector('.tab-btn.active');
            const tabs = Array.from(document.querySelectorAll('.tab-btn'));
            const currentIndex = tabs.indexOf(activeTab);
            
            let nextIndex;
            if (e.key === 'ArrowRight') {
                nextIndex = (currentIndex + 1) % tabs.length;
            } else {
                nextIndex = (currentIndex - 1 + tabs.length) % tabs.length;
            }
            
            tabs[nextIndex].click();
            tabs[nextIndex].focus();
        }
    });

    // Add focus styles for accessibility
    tabBtns.forEach(btn => {
        btn.addEventListener('focus', () => {
            btn.style.outline = '2px solid #5aff5a';
            btn.style.outlineOffset = '2px';
        });
        
        btn.addEventListener('blur', () => {
            btn.style.outline = 'none';
        });
    });
});

// Performance optimization: Reduce particles on mobile
function isMobile() {
    return window.innerWidth <= 768;
}

// Adjust particle frequency based on device
if (isMobile()) {
    // Override the particle creation interval for mobile devices
    document.addEventListener('DOMContentLoaded', function() {
        // Clear existing interval and set a slower one for mobile
        setInterval(() => {
            if (Math.random() > 0.7) { // Only create particles 30% of the time on mobile
                createParticle();
            }
        }, 1000);
    });
} 