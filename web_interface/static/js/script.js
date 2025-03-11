// detection_app/static/detection_app/js/script.js
const themeToggle = document.getElementById('theme-toggle');
const sunIcon = document.getElementById('sun-icon');
const moonIcon = document.getElementById('moon-icon');
const uploadForm = document.getElementById('upload-form');
const uploadStatus = document.getElementById('upload-status');
const resultsSection = document.getElementById('results');
const logContent = document.getElementById('log-content');

themeToggle.addEventListener('click', () => {
    document.documentElement.classList.toggle('dark');
    sunIcon.classList.toggle('hidden');
    moonIcon.classList.toggle('hidden');
    localStorage.setItem('theme', document.documentElement.classList.contains('dark') ? 'dark' : 'light');
});

if (localStorage.getItem('theme') === 'dark') {
    document.documentElement.classList.add('dark');
    sunIcon.classList.remove('hidden');
    moonIcon.classList.add('hidden');
} else {
    sunIcon.classList.add('hidden');
    moonIcon.classList.remove('hidden');
}

uploadForm.addEventListener('submit', (e) => {
    e.preventDefault();
    const formData = new FormData(uploadForm);
    const csrfElement = document.querySelector('[name=csrfmiddlewaretoken]');
    if (!csrfElement) {
        uploadStatus.innerHTML = `<p class="text-red-500">CSRF token missing! 🚫</p>`;
        console.error("CSRF token not found in the form.");
        return;
    }
    const csrfToken = csrfElement.value;
    uploadStatus.innerHTML = '<img src="/static/loading.gif" class="w-8 mx-auto animate-spin" alt="Loading">';
    
    fetch('/upload/', {
        method: 'POST',
        body: formData,
        headers: {
            'X-CSRF-Token': csrfToken
        }
    })
    .then(response => {
        if (!response.ok) throw new Error(`HTTP error! Status: ${response.status}`);
        return response.json();
    })
    .then(data => {
        if (data.status === 'processed' || data.status === 'exists') {
            fetchResults(data.video_name);
            uploadStatus.innerHTML = `<p class="text-green-500">Video ${data.video_name} processed! ✅</p>`;
        } else if (data.status === 'error') {
            uploadStatus.innerHTML = `<p class="text-red-500">Error: ${data.error} 🚨</p>`;
        } else {
            uploadStatus.innerHTML = `<p class="text-red-500">Invalid request 🚫</p>`;
        }
    })
    .catch(error => {
        uploadStatus.innerHTML = `<p class="text-red-500">Fetch error: ${error.message} 🚨</p>`;
        console.error('Fetch error:', error);
    });
});

function fetchResults(videoName) {
    fetch(`/results/${videoName}/`)
        .then(response => {
            if (!response.ok) throw new Error(`HTTP error! Status: ${response.status}`);
            return response.json();
        })
        .then(data => {
            if (data.error) {
                resultsSection.innerHTML = `<p class="text-red-500">Error: ${data.error}</p>`;
                logContent.textContent = 'No logs available.';
                return;
            }
            const result = data.result;
            resultsSection.innerHTML = `
                <div class="bg-white dark:bg-gray-800 p-4 rounded-lg shadow-gold-lg dark:shadow-lg hover:shadow-gold-lg dark:hover:shadow-xl transition transform hover:-translate-y-1 fade-in border border-gold-light dark:border-gray-700">
                    <h3 class="text-xl font-bold text-darkText dark:text-white">${result.video_name} 🎥</h3>
                    <p class="text-2xl ${result.is_fake ? 'text-red-500' : 'text-green-500'}">
                        ${result.is_fake ? 'Fake 🚨' : 'Real ✅'}
                    </p>
                    <p class="text-darkText dark:text-gray-300">Prediction: ${result.pred.toFixed(2)}</p>
                    <p class="text-darkText dark:text-gray-300">Consistency: ${result.consistency.toFixed(2)}</p>
                </div>
            `;
            logContent.textContent = data.log;
        })
        .catch(error => {
            resultsSection.innerHTML = `<p class="text-red-500">Fetch error: ${error.message}</p>`;
            console.error('Fetch error:', error);
        });
}


tailwind.config = {
    darkMode: 'class',
    theme: {
        extend: {
            colors: {
                gold: { light: '#F0D97A', DEFAULT: '#DAA520', dark: '#B8860B' },
                milky: { DEFAULT: '#E0E0E0', dark: '#D0D0D0' },
                darkText: '#333333'
            },
            boxShadow: {
                'gold': '0 4px 6px -1px rgba(218, 165, 32, 0.3), 0 2px 4px -1px rgba(218, 165, 32, 0.2)',
                'gold-lg': '0 10px 15px -3px rgba(218, 165, 32, 0.3), 0 4px 6px -2px rgba(218, 165, 32, 0.2)',
            }
        }
    }
}