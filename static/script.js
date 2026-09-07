/**
 * Customer Churn Predictor - Frontend Controller
 * Interacts with FastAPI /predict endpoint
 */

document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const form = document.getElementById('churnForm');
    const btnSubmit = document.getElementById('btnSubmit');
    const emptyState = document.getElementById('emptyState');
    const analysisContent = document.getElementById('analysisContent');
    const verdictBanner = document.getElementById('verdictBanner');
    const verdictIcon = document.getElementById('verdictIcon');
    const verdictTitle = document.getElementById('verdictTitle');
    const verdictDesc = document.getElementById('verdictDesc');
    const probabilityText = document.getElementById('probabilityText');
    const probabilityBar = document.getElementById('probabilityBar');
    const insightsList = document.getElementById('insightsList');
    const jsonPayload = document.getElementById('jsonPayload');
    const analysisTimestamp = document.getElementById('analysisTimestamp');
    const apiStatusPill = document.getElementById('apiStatusPill');
    const apiStatusText = document.getElementById('apiStatusText');
    const toastContainer = document.getElementById('toastContainer');

    // Preset Buttons
    const btnPresetLoyal = document.getElementById('btnPresetLoyal');
    const btnPresetAverage = document.getElementById('btnPresetAverage');
    const btnPresetAtRisk = document.getElementById('btnPresetAtRisk');
    const btnResetForm = document.getElementById('btnResetForm');

    // Determine API Endpoint (handles file:// protocol or hosted origin)
    const isHosted = window.location.protocol.startsWith('http');
    const API_BASE = isHosted ? '' : 'http://127.0.0.1:8000';
    const PREDICT_URL = `${API_BASE}/predict`;
    const HEALTH_URL = `${API_BASE}/`;

    // Preset Profiles
    const PRESETS = {
        loyal: {
            AccountWeeks: 128,
            ContractRenewal: "1",
            DataPlan: "1",
            DataUsage: 3.8,
            CustServCalls: 0,
            DayMins: 145.2,
            DayCalls: 88,
            MonthlyCharge: 42.50,
            OverageFee: 2.10,
            RoamMins: 6.5
        },
        average: {
            AccountWeeks: 90,
            ContractRenewal: "0",
            DataPlan: "0",
            DataUsage: 0.0,
            CustServCalls: 3,
            DayMins: 200.0,
            DayCalls: 100,
            MonthlyCharge: 62.00,
            OverageFee: 9.50,
            RoamMins: 11.0
        },
        risk: {
            AccountWeeks: 38,
            ContractRenewal: "0",
            DataPlan: "0",
            DataUsage: 0.0,
            CustServCalls: 5,
            DayMins: 295.4,
            DayCalls: 130,
            MonthlyCharge: 89.50,
            OverageFee: 19.80,
            RoamMins: 15.6
        }
    };

    // Check API Health on load
    checkApiHealth();

    async function checkApiHealth() {
        try {
            const res = await fetch(HEALTH_URL, { method: 'GET' });
            if (res.ok) {
                apiStatusPill.className = 'api-status-pill online';
                apiStatusText.textContent = 'API Connected (Ready)';
            } else {
                throw new Error('API returned non-200');
            }
        } catch (err) {
            apiStatusPill.className = 'api-status-pill offline';
            apiStatusText.textContent = 'API Offline (Start FastAPI)';
        }
    }

    // Populate Form helper
    function loadPreset(profile) {
        document.getElementById('AccountWeeks').value = profile.AccountWeeks;
        document.getElementById('DataUsage').value = profile.DataUsage;
        document.getElementById('CustServCalls').value = profile.CustServCalls;
        document.getElementById('DayMins').value = profile.DayMins;
        document.getElementById('DayCalls').value = profile.DayCalls;
        document.getElementById('MonthlyCharge').value = profile.MonthlyCharge;
        document.getElementById('OverageFee').value = profile.OverageFee;
        document.getElementById('RoamMins').value = profile.RoamMins;

        // Radios
        const contractRadio = document.querySelector(`input[name="ContractRenewal"][value="${profile.ContractRenewal}"]`);
        if (contractRadio) contractRadio.checked = true;

        const dataPlanRadio = document.querySelector(`input[name="DataPlan"][value="${profile.DataPlan}"]`);
        if (dataPlanRadio) dataPlanRadio.checked = true;

        showToast('Sample profile data loaded into form.', 'success');
    }

    // Preset Event Listeners
    btnPresetLoyal.addEventListener('click', () => loadPreset(PRESETS.loyal));
    btnPresetAverage.addEventListener('click', () => loadPreset(PRESETS.average));
    btnPresetAtRisk.addEventListener('click', () => loadPreset(PRESETS.risk));

    btnResetForm.addEventListener('click', () => {
        form.reset();
        document.getElementById('contract_yes').checked = true;
        document.getElementById('dataplan_yes').checked = true;
        emptyState.style.display = 'flex';
        analysisContent.style.display = 'none';
        showToast('Form reset.', 'info');
    });

    // Form Submit Handler
    form.addEventListener('submit', async (e) => {
        e.preventDefault();

        // Extract and cast values strictly matching FastAPI pydantic model
        const formData = new FormData(form);

        const payload = {
            AccountWeeks: parseInt(formData.get('AccountWeeks'), 10),
            ContractRenewal: parseInt(formData.get('ContractRenewal'), 10),
            DataPlan: parseInt(formData.get('DataPlan'), 10),
            DataUsage: parseFloat(formData.get('DataUsage')),
            CustServCalls: parseInt(formData.get('CustServCalls'), 10),
            DayMins: parseFloat(formData.get('DayMins')),
            DayCalls: parseInt(formData.get('DayCalls'), 10),
            MonthlyCharge: parseFloat(formData.get('MonthlyCharge')),
            OverageFee: parseFloat(formData.get('OverageFee')),
            RoamMins: parseFloat(formData.get('RoamMins'))
        };

        // Validate types
        for (const [key, value] of Object.entries(payload)) {
            if (Number.isNaN(value) || value === null || value === undefined) {
                showToast(`Please enter a valid value for ${key}.`, 'error');
                return;
            }
        }

        // Set Loading UI
        btnSubmit.classList.add('loading');
        btnSubmit.disabled = true;

        try {
            const startTime = performance.now();
            const response = await fetch(PREDICT_URL, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Accept': 'application/json'
                },
                body: JSON.stringify(payload)
            });

            const elapsed = Math.round(performance.now() - startTime);

            if (!response.ok) {
                const errorData = await response.json().catch(() => null);
                throw new Error(errorData?.detail || `Server responded with status ${response.status}`);
            }

            const result = await response.json();
            
            // Mark API as connected
            apiStatusPill.className = 'api-status-pill online';
            apiStatusText.textContent = 'API Connected';

            // Render Results
            renderResults(result, payload, elapsed);
            showToast('Churn prediction calculated successfully.', 'success');

        } catch (error) {
            console.error('Prediction request error:', error);
            showToast(`Prediction failed: ${error.message}. Is FastAPI server running on http://127.0.0.1:8000?`, 'error');
            apiStatusPill.className = 'api-status-pill offline';
            apiStatusText.textContent = 'API Error / Offline';
        } finally {
            btnSubmit.classList.remove('loading');
            btnSubmit.disabled = false;
        }
    });

    // Render Prediction Output
    function renderResults(res, inputPayload, latencyMs) {
        emptyState.style.display = 'none';
        analysisContent.style.display = 'flex';

        const isChurn = res.prediction === 'Churn';
        const prob = typeof res.churn_probability === 'number' ? res.churn_probability : 0;
        const probPercent = (prob * 100).toFixed(1);

        // Analysis Timestamp
        const now = new Date();
        analysisTimestamp.textContent = `${now.toLocaleTimeString()} (${latencyMs}ms)`;

        // Verdict Banner
        if (isChurn) {
            verdictBanner.className = 'verdict-banner churn';
            verdictTitle.textContent = 'High Risk of Churn';
            verdictDesc.textContent = 'Model flags high likelihood of customer contract termination.';
            verdictIcon.innerHTML = `
                <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
                    <circle cx="12" cy="12" r="10"></circle>
                    <line x1="12" y1="8" x2="12" y2="12"></line>
                    <line x1="12" y1="16" x2="12.01" y2="16"></line>
                </svg>
            `;
        } else {
            verdictBanner.className = 'verdict-banner safe';
            verdictTitle.textContent = 'Retained (No Churn)';
            verdictDesc.textContent = 'Customer demonstrates healthy engagement and high retention odds.';
            verdictIcon.innerHTML = `
                <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
                    <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path>
                    <polyline points="22 4 12 14.01 9 11.01"></polyline>
                </svg>
            `;
        }

        // Probability Counter & Bar Animation
        animateValue(probabilityText, 0, parseFloat(probPercent), 600, '%');
        
        probabilityBar.style.width = `${Math.min(Math.max(probPercent, 4), 100)}%`;
        if (prob < 0.35) {
            probabilityBar.className = 'progress-bar-fill safe';
        } else if (prob < 0.60) {
            probabilityBar.className = 'progress-bar-fill medium';
        } else {
            probabilityBar.className = 'progress-bar-fill churn';
        }

        // Generate Rule-Based Risk Indicators
        generateRiskIndicators(inputPayload, isChurn, prob);

        // Populate JSON Debugger
        jsonPayload.textContent = JSON.stringify({
            request_payload: inputPayload,
            response: res,
            endpoint: PREDICT_URL
        }, null, 2);
    }

    // Rule-Based Customer Risk Indicators Evaluator
    function generateRiskIndicators(data, isChurn, prob) {
        insightsList.innerHTML = '';
        const insights = [];

        if (data.CustServCalls >= 4) {
            insights.push({
                type: 'danger',
                text: `<strong>High Support Activity:</strong> Customer has made ${data.CustServCalls} customer service calls, signaling dissatisfaction.`
            });
        } else if (data.CustServCalls === 0) {
            insights.push({
                type: 'good',
                text: '<strong>Zero Support Complaints:</strong> No customer service calls recorded.'
            });
        }

        if (data.ContractRenewal === 0) {
            insights.push({
                type: 'warn',
                text: '<strong>Unrenewed Contract:</strong> The customer has not renewed their contract recently.'
            });
        } else {
            insights.push({
                type: 'good',
                text: '<strong>Active Renewal:</strong> Customer has recently committed to contract renewal.'
            });
        }

        if (data.OverageFee > 15.0) {
            insights.push({
                type: 'danger',
                text: `<strong>Excess Overage Charges:</strong> Peak overage fee of $${data.OverageFee.toFixed(2)} may cause bill shock.`
            });
        }

        if (data.AccountWeeks > 100) {
            insights.push({
                type: 'good',
                text: `<strong>High Tenure:</strong> Account active for ${data.AccountWeeks} weeks (~${(data.AccountWeeks/52).toFixed(1)} years).`
            });
        }

        if (insights.length === 0) {
            insights.push({
                type: isChurn ? 'warn' : 'good',
                text: isChurn ? 'Usage patterns correlate with churn risk based on aggregated feature weights.' : 'Overall telemetry reflects standard customer usage pattern.'
            });
        }

        insights.forEach(item => {
            const li = document.createElement('li');
            li.className = `insight-item ${item.type}`;
            const iconSvg = item.type === 'good' 
                ? `<svg class="insight-icon" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="20 6 9 17 4 12"/></svg>`
                : item.type === 'danger'
                ? `<svg class="insight-icon" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>`
                : `<svg class="insight-icon" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>`;
            
            li.innerHTML = `${iconSvg}<div>${item.text}</div>`;
            insightsList.appendChild(li);
        });
    }

    // Number animation utility
    function animateValue(obj, start, end, duration, suffix = '') {
        let startTimestamp = null;
        const step = (timestamp) => {
            if (!startTimestamp) startTimestamp = timestamp;
            const progress = Math.min((timestamp - startTimestamp) / duration, 1);
            const val = (progress * (end - start) + start).toFixed(1);
            obj.innerHTML = `${val}${suffix}`;
            if (progress < 1) {
                window.requestAnimationFrame(step);
            }
        };
        window.requestAnimationFrame(step);
    }

    // Toast Notification helper
    function showToast(message, type = 'info') {
        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        
        let iconSvg = '';
        if (type === 'error') {
            iconSvg = `<svg class="toast-icon" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/></svg>`;
        } else if (type === 'success') {
            iconSvg = `<svg class="toast-icon" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="20 6 9 17 4 12"/></svg>`;
        } else {
            iconSvg = `<svg class="toast-icon" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/></svg>`;
        }

        toast.innerHTML = `${iconSvg}<span>${message}</span>`;
        toastContainer.appendChild(toast);

        setTimeout(() => {
            toast.style.opacity = '0';
            toast.style.transform = 'translateY(10px)';
            toast.style.transition = 'all 0.3s ease';
            setTimeout(() => toast.remove(), 300);
        }, 4000);
    }
});
