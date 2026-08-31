(function () {
    'use strict';

    const API_BASE = 'http://localhost:5000';
    const state = {
        dataSource: 'offline',
        dataCount: 0,
        updatedAt: null,
        apn: 'checking',
        llm: 'checking'
    };

    const labels = {
        source: {
            offline: '离线/缺失',
            csv: 'CSV 导入',
            realtime: '实时数据',
            simulation: '模拟数据',
            prediction: '预测数据',
            unknown: '未知'
        },
        apn: {
            checking: '检测中',
            real: '真实 APN',
            fallback: 'Fallback',
            offline: '离线'
        },
        llm: {
            checking: '检测中',
            configured: '已配置',
            fallback: 'Fallback',
            unavailable: '不可用',
            offline: '离线'
        }
    };

    function nowText(value) {
        const date = value ? new Date(value) : new Date();
        if (Number.isNaN(date.getTime())) return '--';
        return date.toLocaleString('zh-CN', {
            month: '2-digit',
            day: '2-digit',
            hour: '2-digit',
            minute: '2-digit',
            second: '2-digit'
        });
    }

    function ensureContainer() {
        let node = document.querySelector('.data-status-container');
        if (node) return node;

        node = document.createElement('section');
        node.className = 'data-status-container';
        node.innerHTML = [
            '<div class="data-status-item data-status-source">',
            '  <i class="fas fa-database"></i>',
            '  <span class="data-status-label">数据来源</span>',
            '  <strong class="data-status-source-text">离线/缺失</strong>',
            '</div>',
            '<div class="data-status-item">',
            '  <i class="fas fa-clock"></i>',
            '  <span class="data-status-label">更新时间</span>',
            '  <strong class="data-status-time-text">--</strong>',
            '</div>',
            '<div class="data-status-item data-status-apn">',
            '  <i class="fas fa-brain"></i>',
            '  <span class="data-status-label">APN</span>',
            '  <strong class="data-status-apn-text">检测中</strong>',
            '</div>',
            '<div class="data-status-item data-status-llm">',
            '  <i class="fas fa-comment-dots"></i>',
            '  <span class="data-status-label">LLM</span>',
            '  <strong class="data-status-llm-text">检测中</strong>',
            '</div>',
            '<div class="data-status-item">',
            '  <i class="fas fa-table"></i>',
            '  <span class="data-status-label">记录数</span>',
            '  <strong class="data-status-count-text">0 条</strong>',
            '</div>'
        ].join('');

        const header = document.querySelector('.page-header, .header, header');
        if (header && header.parentNode) {
            header.insertAdjacentElement('afterend', node);
            return node;
        }

        const host = document.querySelector('.main-content, .page-container, .container, .app-container') || document.body;
        host.insertAdjacentElement(host === document.body ? 'afterbegin' : 'afterbegin', node);
        return node;
    }

    function normalizeSource(raw) {
        const value = String(raw || '').toLowerCase();
        if (value.includes('csv') || value.includes('import')) return 'csv';
        if (value.includes('real') || value.includes('live')) return 'realtime';
        if (value.includes('sim')) return 'simulation';
        if (value.includes('predict')) return 'prediction';
        if (value.includes('offline') || value.includes('missing')) return 'offline';
        return raw ? 'unknown' : 'offline';
    }

    function readSharedData() {
        const aqua = window.AquaMindData;
        if (!aqua) return;

        const meta = typeof aqua.getMeta === 'function' ? aqua.getMeta() : aqua.meta;
        const data = typeof aqua.getData === 'function' ? aqua.getData() : aqua.data;
        const source = meta && (meta.source || meta.dataSource || meta.mode);

        if (Array.isArray(data)) state.dataCount = data.length;
        if (meta && Number.isFinite(meta.count)) state.dataCount = meta.count;
        if (meta && meta.updatedAt) state.updatedAt = meta.updatedAt;
        if (meta && meta.lastUpdated) state.updatedAt = meta.lastUpdated;
        state.dataSource = normalizeSource(source);
    }

    function classifyApn(health) {
        const service = health && health.services && health.services.apn_model;
        if (!service) return 'fallback';
        const status = String(service.status || '').toLowerCase();
        if (service.model_loaded === true || status.includes('loaded') || status.includes('real')) return 'real';
        if (status.includes('fallback') || service.fallback_available === true) return 'fallback';
        return 'offline';
    }

    function classifyLlm(health, info) {
        const service = health && health.services && health.services.llm;
        if (info && (info.api_configured === true || info.configured === true)) return 'configured';
        if (service && (service.api_configured === true || service.configured === true)) return 'configured';
        if ((info && info.fallback_enabled === true) || (service && service.fallback_enabled === true)) return 'fallback';
        if (service && String(service.status || '').toLowerCase().includes('fallback')) return 'fallback';
        return 'unavailable';
    }

    async function fetchJson(url) {
        const response = await fetch(url, { cache: 'no-store' });
        if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
        return response.json();
    }

    async function refreshBackendStatus() {
        try {
            const health = await fetchJson(`${API_BASE}/api/health`);
            let info = null;
            try {
                info = await fetchJson(`${API_BASE}/api/chat/info`);
            } catch (error) {
                info = null;
            }
            state.apn = classifyApn(health);
            state.llm = classifyLlm(health, info);
        } catch (error) {
            state.apn = 'offline';
            state.llm = 'offline';
        }
        render();
    }

    function setStatusClass(node, prefix, value) {
        node.className = node.className
            .split(/\s+/)
            .filter(Boolean)
            .filter((item) => !item.startsWith(`${prefix}-`))
            .join(' ');
        node.classList.add(`${prefix}-${value}`);
    }

    function render() {
        const node = ensureContainer();
        readSharedData();

        node.querySelector('.data-status-source-text').textContent = labels.source[state.dataSource] || labels.source.unknown;
        node.querySelector('.data-status-time-text').textContent = nowText(state.updatedAt);
        node.querySelector('.data-status-apn-text').textContent = labels.apn[state.apn] || labels.apn.offline;
        node.querySelector('.data-status-llm-text').textContent = labels.llm[state.llm] || labels.llm.unavailable;
        node.querySelector('.data-status-count-text').textContent = `${state.dataCount || 0} 条`;

        setStatusClass(node.querySelector('.data-status-source'), 'source', state.dataSource);
        setStatusClass(node.querySelector('.data-status-apn'), 'apn', state.apn);
        setStatusClass(node.querySelector('.data-status-llm'), 'llm', state.llm);
    }

    function init() {
        render();
        refreshBackendStatus();
        window.addEventListener('aquamind:data-ready', render);
        window.addEventListener('aquamind:data-updated', render);
        window.addEventListener('storage', render);
        window.setInterval(refreshBackendStatus, 30000);
    }

    window.DataStatus = {
        refresh: function () {
            render();
            return refreshBackendStatus();
        },
        getState: function () {
            return Object.assign({}, state);
        }
    };

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();
