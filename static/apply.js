const form = document.getElementById('loan-form');
const submitBtn = document.getElementById('submit-btn');
const resultBox = document.getElementById('result-box');
const resultIcon = document.getElementById('result-icon');
const resultText = document.getElementById('result-text');
const resultProb = document.getElementById('result-prob');
const errorBox = document.getElementById('error-box');

form.addEventListener('submit', async (e) => {
  e.preventDefault();

  errorBox.style.display = 'none';
  resultBox.classList.remove('show', 'approved', 'rejected');
  submitBtn.disabled = true;
  submitBtn.textContent = 'Checking...';

  const formData = new FormData(form);
  const payload = Object.fromEntries(formData.entries());

  try {
    const res = await fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'Prediction failed');

    resultBox.classList.add('show', data.approved ? 'approved' : 'rejected');
    resultIcon.textContent = data.approved ? '✅' : '❌';
    resultText.textContent = data.approved ? 'Loan approved' : 'Loan not approved';
    resultProb.textContent =
      data.approval_probability !== undefined
        ? `${(data.approval_probability * 100).toFixed(1)}% approval likelihood`
        : '';
  } catch (err) {
    errorBox.textContent = err.message || 'Something went wrong. Please try again.';
    errorBox.style.display = 'block';
  } finally {
    submitBtn.disabled = false;
    submitBtn.textContent = 'Check loan eligibility';
  }
});
