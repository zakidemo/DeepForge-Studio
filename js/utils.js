export const utils = {
    debounce(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    },

    sanitizeHTML(str) {
        const temp = document.createElement('div');
        temp.textContent = str;
        return temp.innerHTML;
    },

    escapeHTML(unsafe) {
        return unsafe
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#039;");
    },

    notify(message, type = 'info') {
        const notification = document.createElement('div');
        notification.className = `notification ${type}`;
        const icon = document.createElement('i');
        icon.className = 'fas fa-info-circle';
        icon.setAttribute('aria-hidden', 'true');
        const text = document.createElement('span');
        text.textContent = ` ${message}`;
        notification.appendChild(icon);
        notification.appendChild(text);
        notification.setAttribute('role', 'alert');
        notification.setAttribute('aria-live', 'polite');
        
        document.body.appendChild(notification);
        
        setTimeout(() => {
            notification.style.animation = 'fadeOut 0.3s ease';
            setTimeout(() => notification.remove(), 300);
        }, 3000);
    },

    copyToClipboard(text) {
        return navigator.clipboard.writeText(text)
            .then(() => this.notify('Copied to clipboard!', 'success'))
            .catch(err => this.notify('Failed to copy', 'error'));
    },

    isParamEnabled(paramName) {
        const toggle = document.getElementById(`toggle_${paramName}`);
        return toggle ? toggle.checked : true;
    },

    getEnabledFeatures() {
        const features = [];
        if (this.isParamEnabled('lrSchedule')) features.push('LR Schedule');
        if (features.length === 0) features.push('Basic Training');
        return features;
    },

    shakeElement(elementId) {
        const element = document.getElementById(elementId);
        if (element) {
            element.style.animation = 'shake 0.5s';
            setTimeout(() => element.style.animation = '', 500);
        }
    },

    highlightElement(element) {
        element.style.transition = 'all 0.3s ease';
        element.style.boxShadow = '0 0 20px rgba(80, 250, 123, 0.5)';
        element.style.borderColor = '#50fa7b';
        
        setTimeout(() => {
            element.style.boxShadow = '';
            element.style.borderColor = '';
        }, 2000);
    },

    showLoadingOverlay(message = 'Processing...') {
        const overlay = document.getElementById('ai-loading');
        if (overlay) {
            document.getElementById('loading-message').textContent = message;
            overlay.style.display = 'flex';
        }
    },

    hideLoadingOverlay() {
        const overlay = document.getElementById('ai-loading');
        if (overlay) {
            overlay.style.display = 'none';
        }
    },

    downloadBlob(blob, filename) {
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }
};