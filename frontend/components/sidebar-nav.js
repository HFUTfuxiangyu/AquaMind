/**
 * 智慧水务系统 - 左侧导航组件
 * 动态生成导航栏，支持响应式和 active 状态
 */

(function() {
    'use strict';

    // 导航配置
    const navConfig = [
        {
            section: '核心功能',
            items: [
                { id: 'dashboard', text: '智能驾驶舱', icon: 'fa-gauge-high', url: 'smart_water_system.html', order: 1 },
                { id: 'map', text: '工艺地理图', icon: 'fa-map-marked-alt', url: 'process_map.html', order: 2 },
                { id: 'warning', text: '水质预警', icon: 'fa-triangle-exclamation', url: 'smart_water_system_v2.html', order: 3 }
            ]
        },
        {
            section: '优化管理',
            items: [
                { id: 'dosage', text: 'AI 加药优化', icon: 'fa-flask', url: 'ai_dosage.html', order: 4 },
                { id: 'device', text: '设备健康', icon: 'fa-heart-pulse', url: 'device_health.html', order: 5 },
                { id: 'energy', text: '能效调度', icon: 'fa-bolt', url: 'energy_schedule.html', order: 6 }
            ]
        },
        {
            section: '数据分析',
            items: [
                { id: 'insight', text: '数据洞察', icon: 'fa-chart-pie', url: 'data_insight.html', order: 7 }
            ]
        }
    ];

    // 获取当前页面 URL
    function getCurrentPage() {
        const path = window.location.pathname;
        const filename = path.substring(path.lastIndexOf('/') + 1).toLowerCase();
        return filename || 'smart_water_system.html';
    }

    // 生成导航 HTML
    function generateNavHTML() {
        const currentPage = getCurrentPage();
        let html = `
            <div class="sidebar">
                <div class="sidebar-header">
                    <div class="sidebar-logo">
                        <div class="sidebar-logo-icon">
                            <i class="fas fa-droplet"></i>
                        </div>
                        <div class="sidebar-logo-text">AquaMind 智慧水务</div>
                    </div>
                </div>
                <div class="sidebar-nav">`;

        navConfig.forEach(section => {
            html += `
                <div class="nav-section">
                    <div class="nav-section-title">${section.section}</div>`;

            section.items.forEach(item => {
                const isActive = currentPage === item.url.toLowerCase();
                const activeClass = isActive ? 'active' : '';
                html += `
                    <a href="${item.url}" class="nav-item ${activeClass}" data-page="${item.id}">
                        <div class="nav-item-icon">
                            <i class="fas ${item.icon}"></i>
                        </div>
                        <div class="nav-item-text">${item.text}</div>
                    </a>`;
            });

            html += `</div>`;
        });

        html += `
                </div>
                <div class="sidebar-footer">
                    <div class="sidebar-status">
                        <div class="status-indicator"></div>
                        <span>系统运行正常</span>
                    </div>
                </div>
            </div>
            <div class="sidebar-overlay" id="sidebarOverlay"></div>`;

        return html;
    }

    // 生成顶部工具栏 HTML
    function generateTopBarHTML(title) {
        return `
            <div class="top-bar">
                <div style="display: flex; align-items: center; gap: 8px;">
                    <button class="menu-toggle" id="menuToggle" title="切换侧边栏">
                        <i class="fas fa-bars"></i>
                    </button>
                    <div class="page-title">${title}</div>
                </div>
                <div class="page-actions">
                    <div class="toolbar-group">
                        <span style="font-size: 12px; color: var(--text-secondary);">
                            <i class="far fa-clock"></i>
                            <span id="systemClock">--:--</span>
                        </span>
                    </div>
                </div>
            </div>`;
    }

    // 初始化导航
    function initNavigation() {
        // 创建导航容器（如果不存在）
        if (!document.querySelector('.app-wrapper')) {
            const body = document.body;
            const existingContent = body.innerHTML;

            // 获取页面标题
            const pageTitle = document.querySelector('h1')?.textContent ||
                            document.title.replace('智慧水务', '').replace('| AquaMind Pro', '').trim() ||
                            '智能驾驶舱';

            // 创建新的布局结构
            body.innerHTML = `
                <div class="app-wrapper">
                    ${generateNavHTML()}
                    <div class="main-content">
                        ${generateTopBarHTML(pageTitle)}
                        <div class="content-area">
                            ${existingContent}
                        </div>
                    </div>
                </div>
            `;
        }

        // 绑定事件
        bindEvents();
        // 更新时钟
        updateClock();
        setInterval(updateClock, 1000);
    }

    // 绑定事件
    function bindEvents() {
        const menuToggle = document.getElementById('menuToggle');
        const sidebar = document.querySelector('.sidebar');
        const overlay = document.getElementById('sidebarOverlay');

        if (menuToggle && sidebar) {
            menuToggle.addEventListener('click', function() {
                if (window.innerWidth <= 768) {
                    sidebar.classList.toggle('mobile-open');
                    overlay.classList.toggle('active');
                } else {
                    sidebar.classList.toggle('collapsed');
                }
            });
        }

        if (overlay) {
            overlay.addEventListener('click', function() {
                sidebar.classList.remove('mobile-open');
                overlay.classList.remove('active');
            });
        }

        // 响应式处理
        window.addEventListener('resize', function() {
            if (window.innerWidth > 768) {
                sidebar.classList.remove('mobile-open');
                overlay.classList.remove('active');
            }
        });
    }

    // 更新时钟
    function updateClock() {
        const clockEl = document.getElementById('systemClock');
        if (clockEl) {
            const now = new Date();
            clockEl.textContent = now.toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' });
        }
    }

    // 更新导航 active 状态（供其他页面调用）
    function updateActiveNav(pageId) {
        document.querySelectorAll('.nav-item').forEach(item => {
            item.classList.remove('active');
            if (item.dataset.page === pageId) {
                item.classList.add('active');
            }
        });
    }

    // 暴露全局方法
    window.SidebarNav = {
        init: initNavigation,
        updateActive: updateActiveNav,
        generateNavHTML: generateNavHTML,
        generateTopBarHTML: generateTopBarHTML
    };

    // DOM 加载完成后自动初始化
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initNavigation);
    } else {
        initNavigation();
    }

})();
